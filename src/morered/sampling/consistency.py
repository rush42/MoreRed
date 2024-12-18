from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from schnetpack import properties
from torch import nn
from tqdm import tqdm

from morered.processes import DiffusionProcess
from morered.sampling import Sampler
from morered.sampling.morered import MoreRedAS
from morered.utils import compute_neighbors, scatter_mean

__all__ = ["ConsistencySampler"]

class ConsistencySampler(MoreRedAS):
    """
    Implements the adaptive Consistency denoiser.
    """

    def __init__(
        self,
        diffusion_process: DiffusionProcess,
        mean_predictor: Union[str, nn.Module],
        time_predictor: Union[None, str, nn.Module] = None,
        time_key: str = "t",
        time_pred_key: str = "t_pred",
        mean_pred_key: str = "mu_pred",
        cutoff: float = 5.0,
        recompute_neighbors: bool = False,
        save_progress: bool = False,
        progress_stride: int = 1,
        results_on_cpu: bool = True,
        device: Optional[torch.device] = None,
        convergence_step: int = 0,
    ):
        """
        Args:
            diffusion_process: The diffusion processe to sample the target property.
            mean_predictor: mean predictor or path to mean predictor to use for the reverse process.
            time_predictor: Seperate diffusion time step predictor or path to the model.
                            Used for 'MoreRed-ITP' and 'MoreRed-AS'.
        """
        self.diffusion_process = diffusion_process
        self.mean_predictor = mean_predictor
        self.time_predictor = time_predictor
        self.mean_pred_key = mean_pred_key
        self.time_pred_key = time_pred_key
        self.convergence_step = convergence_step
        self.cutoff = cutoff
        self.device = device
        self.time_key = time_key
        self.recompute_neighbors = recompute_neighbors
        self.save_progress = save_progress
        self.progress_stride = progress_stride
        self.results_on_cpu = results_on_cpu

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.mean_predictor, str):
            self.mean_predictor = torch.load(self.mean_predictor, map_location=self.device).eval()
        elif self.mean_predictor is not None:
            self.mean_predictor = self.mean_predictor.to(self.device).eval()
        
        if  self.time_predictor is not None:
            if isinstance(self.time_predictor, str):
                self.time_predictor = torch.load(
                    self.time_predictor, device=self.device
                ).eval()
            else:
                self.time_predictor = self.time_predictor.to(self.device).eval()


    @torch.no_grad()
    def inference_step(
        self, inputs: Dict[str, torch.Tensor], time_steps
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get one time step per molecule
        and the noise prediction.

        Args:
            inputs: input data for noise prediction.
            iter: the current iteration of the reverse process.
        """


        # append the normalized time step to the model input
        # We first unnormlize the time steps to get a binned step as during training
        inputs[self.time_key] = self.diffusion_process.normalize_time(time_steps)
        inputs[self.time_key] = inputs[self.time_key][inputs[properties.idx_m]]

        # cast input to float for the denoiser
        for key, val in inputs.items():
            if val.dtype == torch.float64:
                inputs[key] = val.float()

        # fetch the noise prediction
        model_out = self.mean_predictor(inputs)
        mean_pred = model_out[self.mean_pred_key].detach()

        return mean_pred

    def sample(self, inputs:  Dict[str, torch.Tensor], t: int = None, inplace: bool = False) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Sample from the consistency model with a single forward pass
        
        Args:
            inputs: dict with input data in the SchNetPack form,
                    including the starting x_t.
        """
        batch = self.prepare_batch(inputs, inplace=inplace)

        if t is None:
            t = self.get_time_steps(batch)
        else:
            t = torch.full_like(inputs[properties.n_atoms], fill_value=t, device=self.device)        


        hist = [
            {
                properties.R: inputs[properties.R].cpu().float().clone(),
                self.time_key: t
            }
        ]

        # get the time predicted mean from the consistency denoiser
        x_0 = {
            properties.R: self.inference_step(batch, t)
        }

        hist.append({
                properties.R: inputs[properties.R].cpu().float().clone(),
                self.time_key: torch.full_like(t, fill_value=0)
            }
        )

        
        return x_0, torch.tensor(1), hist

    @torch.no_grad()
    def denoise(
        self,
        inputs: Dict[str, torch.Tensor],
        t: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, torch.Tensor]]]:
        
        return self.sample(inputs, t)


    @torch.no_grad()
    def sample_ms(
        self,
        inputs: Dict[str, torch.Tensor],
        iters: Optional[int] = 1,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Peforns denoising/sampling using the adaptive reverse diffusion process.
        Returns the denoised/sampled data, the number of steps taken,
        and the progress history if saved.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting x_t.
            max_steps: the maximum number of reverse steps to perform.
        """

        batch = self.prepare_batch(inputs)
        idx_m = batch[properties.idx_m]
        
        # initialize convergence flag for each molecule
        converged = torch.zeros_like(
            batch[properties.n_atoms], dtype=torch.bool, device=self.device
        )

        # initialize the number of steps taken for each molecule
        num_steps = torch.full_like(
            batch[properties.n_atoms], -1, dtype=torch.long, device=self.device
        )

        noise = torch.zeros_like(batch[properties.R], device=self.device)
        # history of the reverse steps
        hist = []

        # simulate the reverse process
        iter = 0
        pbar = tqdm()
        while iter < iters:
            # update the neighbors list if required
            if self.recompute_neighbors:
                batch = compute_neighbors(batch, cutoff=self.cutoff, device=self.device)

            batch[properties.R] += noise

            t = self.get_time_steps(batch, iter)
            # get the time steps and noise predictions from the denoiser
            mean = self.inference_step(batch, t)

            # # save history if required. Must be done before the reverse step.
            # if self.save_progress and (
            #     iter % self.progress_stride == 0 or iter == iters - 1
            # ):
            #     hist.append(
            #         {
            #             properties.R: batch[properties.R].cpu().float().clone(),
            #             self.time_key: time_steps.cpu(),
            #         }
            #     )

            # perform one reverse step

            # update only non-converged molecules.
            mask_converged = converged[idx_m]
            batch[properties.R] = (
                mask_converged.unsqueeze(-1) * batch[properties.R]
                + (~mask_converged).unsqueeze(-1) * mean
            )

            # use the average time step for convergence check
            converged = converged | (t <= self.convergence_step)

            iter += 1
            pbar.update(1)

            # save the number of steps
            num_steps[converged & (num_steps < 0)] = iter

            # check if all molecules converged and end the denoising
            if converged.all():
                break
            
            # t = torch.full_like(t, fill_value=1)
            # # diffuse data for next iteration
            # diffused, _ = self.diffusion_process.diffuse(batch[properties.R], idx_m, t[idx_m])
            # batch[properties.R][~mask_converged, :] = diffused[~mask_converged, :]

        pbar.close()

        # prepare the final output
        x_0 = {
            properties.R: (
                batch[properties.R].cpu()
                if self.results_on_cpu
                else batch[properties.R]
            )
        }

        num_steps = num_steps.cpu()

        return x_0, num_steps, hist
