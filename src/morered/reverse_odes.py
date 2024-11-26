from abc import abstractmethod
from typing import Dict, List, Tuple, Union, Optional

import torch
from schnetpack import properties
from torch import nn
from tqdm import tqdm

from morered.processes import DiffusionProcess
from morered.utils import compute_neighbors

__all__ = ["DDPM"]


class ReverseODE:
    """
    Implements the plain DDPM ancestral sampler proposed by Ho et al. 2020
    Subclasses the base class 'Sampler'.
    """

    def __init__(
        self,
        diffusion_process: DiffusionProcess,
        denoiser: Union[str, nn.Module],
        time_key: str = "t",
        noise_pred_key: str = "eps_pred",
        **kwargs,
    ):
        """
        Args:
            diffusion_process: The diffusion processe to sample the target property.
            denoiser: denoiser or path to denoiser to use for the reverse process.
            time_key: the key for the time.
            noise_pred_key: the key for the noise prediction.
        """
        super().__init__(diffusion_process, denoiser, **kwargs)
        self.time_key = time_key
        self.noise_pred_key = noise_pred_key

    @abstractmethod
    def get_increment(
        self, batch: Dict[str, torch.Tensor], t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the increment for the reverse ODE process.
        Args:
            batch: dict with input data in the SchNetPack form
            i: the time step of the reverse process.
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_time_steps(
        self, inputs: Dict[str, torch.Tensor], iter: int
    ) -> torch.Tensor:
        # current reverse time step
        time_steps = torch.full_like(
            inputs[properties.n_atoms],
            fill_value=iter,
            dtype=torch.long,
            device=self.device,
        )
        return time_steps

    @torch.no_grad()
    def inference_step(
        self, inputs: Dict[str, torch.Tensor], t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get the time steps and noise prediction.

        Args:
            inputs: input data for noise prediction.
            iter: the current iteration of the reverse process.
        """

        # append the normalized time step to the model input
        inputs[self.time_key] = self.diffusion_process.normalize_time(t)

        # broadcast the time step to atoms-level
        inputs[self.time_key] = inputs[self.time_key][inputs[properties.idx_m]]

        # cast input to float for the denoiser
        for key, val in inputs.items():
            if val.dtype == torch.float64:
                inputs[key] = val.float()

        # forward pass through the denoiser
        model_out = self.denoiser(inputs)

        # fetch the noise prediction
        noise_pred = model_out[self.noise_pred_key].detach()

        return noise_pred

    def denoise(
        self,
        inputs: Dict[str, torch.Tensor],
        t: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Peforns denoising/sampling using the estimated score function with an ODE solver.
        Returns the denoised/sampled data, the number of steps taken,
        and the progress history if saved.

        Args:
            inputs: dict with input data in the SchNetPack form,
                    inluding the starting x_t.
            t: the start step of the reverse process, between 1 and T.
                If None, set t = T.
        """
        # Default is t=T
        if t is None:
            t = self.diffusion_process.get_T()

        if not isinstance(t, int) or t < 1 or t > self.diffusion_process.get_T():
            raise ValueError(
                "t must be one int between 1 and T that indicates the starting step."
                "Sampling using different starting steps is not supported yet for DDPM."
            )

        batch = self.prepare_batch(inputs)

        hist = []

        # simulate the reverse process
        for i in tqdm(range(t - 1, -1, -1)):
            # update the neighbors list if required
            if self.recompute_neighbors:
                batch = compute_neighbors(batch, cutoff=self.cutoff, device=self.device)
            
                    # current reverse time step
            time_steps = self.get_time_steps(inputs, i)

            # get the time steps and noise predictions from the denoiser
            increment = self.get_increment(batch, time_steps)

            # save history if required. Must be done before the reverse step.
            if self.save_progress and (i % self.progress_stride == 0):
                hist.append(
                    {
                        properties.R: batch[properties.R].cpu().float().clone(),
                        self.time_key: time_steps.cpu(),
                    }
                )

            # update the state
            batch[properties.R] -= increment

        # prepare the final output
        x_0 = {
            properties.R: (
                batch[properties.R].cpu()
                if self.results_on_cpu
                else batch[properties.R]
            )
        }

        num_steps = torch.full_like(
            batch[properties.n_atoms], t, dtype=torch.long, device="cpu"
        )

        return x_0, num_steps, hist


class ReverseODEEuler(ReverseODE):
    """
    Implements a 'Reverse ODE' using Euler's method.
    """

    def __init__(self, *args, **kwargs):
        # Pass all arguments to the `DDPM` constructor
        super().__init__(*args, **kwargs)

    def get_increment(
        self, batch: Dict[str, torch.Tensor], time_steps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        mask = time_steps == 0

        # get the time steps and noise predictions from the denoiser
        noise = self.inference_step(batch, time_steps)

        scores = self.diffusion_process.score_function(
            noise, batch[properties.idx_m], time_steps[batch[properties.idx_m]]
        )

        scores[mask, :] = 0

        # perform one reverse step
        return scores


class ReverseODEHeun(ReverseODE):
    """
    Implements a 'Reverse ODE' using Heun's method.
    """

    def __init__(self, *args, **kwargs):
        # Pass all arguments to the `DDPM` constructor
        super().__init__(*args, **kwargs)

    def get_increment(
        self, batch: Dict[str, torch.Tensor], time_steps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # get the noise predictions from the denoiser and calculate the scores
        noise = self.inference_step(batch, time_steps)
        scores = self.diffusion_process.score_function(
            noise, batch[properties.idx_m], time_steps[batch[properties.idx_m]]
        )

        # perform one reverse step to get the intermediate state
        intermediate_state = batch.copy()
        intermediate_state[properties.R] = batch[properties.R] - scores
        intermediate_time_steps = time_steps.copy()
        intermediate_time_steps[time_steps != 0] -= 1

        # get the noise predictions for the intermediate state
        intermediate_noise = self.inference_step(
            intermediate_state, time_steps - 1
        )

        intermediate_scores = self.diffusion_process.score_function(
            intermediate_noise,
            batch[properties.idx_m],
            intermediate_time_steps[batch[properties.idx_m]],
        )

        increment = 0.5 * (scores + intermediate_scores)
        increment[time_steps == 1] = scores
        increment[time_steps == 0, :] = 0

        # update the batch with the averaged scores
        return increment
