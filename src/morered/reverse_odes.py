from abc import abstractmethod
from typing import Dict, List, Tuple, Union, Optional

import torch
from schnetpack import properties
from torch import nn
from tqdm import tqdm

from morered.processes import DiffusionProcess
from morered.utils import compute_neighbors, scatter_mean

__all__ = ["ReverseODE", "ReverseODEEuler", "ReverseODEHeun"]


class ReverseODE:
    """
    Abstract Class for ReverseODEs.
    """

    def __init__(
        self,
        diffusion_process: DiffusionProcess,
        denoiser: Union[str, nn.Module],
        time_key: str = "t",
        noise_pred_key: str = "eps_pred",
        recompute_neighbors: bool = False,
        save_progress: bool = False,
        progress_stride: int = 1,
        results_on_cpu: bool = True,
        device: Optional[torch.device] = None,
        cutoff: Optional[float] = None,
    ):
        """
        Args:
            diffusion_process: The diffusion processe to sample the target property.
            denoiser: denoiser or path to denoiser to use for the reverse process.
            time_key: the key for the time.
            noise_pred_key: the key for the noise prediction.
        """
        self.diffusion_process = diffusion_process
        self.denoiser = denoiser
        self.time_key = time_key
        self.noise_pred_key = noise_pred_key
        self.recompute_neighbors = recompute_neighbors
        self.save_progress = save_progress
        self.progress_stride = progress_stride
        self.results_on_cpu = results_on_cpu
        self.device = device
        self.cutoff = cutoff

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(denoiser, str):
            self.denoiser = torch.load(self.denoiser, map_location=self.device).eval()
        elif self.denoiser is not None:
            self.denoiser = self.denoiser.to(self.device).eval()

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

    def prepare_batch(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # copy inputs to avoid inplace operations
        batch = {prop: val.clone().to(self.device) for prop, val in inputs.items()}

        # check if center of geometry is close to zero
        CoG = scatter_mean(
            batch[properties.R], batch[properties.idx_m], batch[properties.n_atoms]
        )
        if self.diffusion_process.invariant and (CoG > 1e-5).any():
            raise ValueError(
                "The input positions are not centered, "
                "while the specified diffusion process is invariant."
            )

        # set all atoms as neighbors and compute neighbors only once before starting.
        if not self.recompute_neighbors:
            batch = compute_neighbors(batch, fully_connected=True, device=self.device)

        return batch

    @torch.no_grad()
    def inference_step(
        self, inputs: Dict[str, torch.Tensor], t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get the time steps and noise prediction.

        Args:
            inputs: input data for noise prediction.
            t: the time step for each atom.
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
        # broadcast the time step to atoms-level
        time_steps = time_steps[batch[properties.idx_m]]

        # get the time steps and noise predictions from the denoiser
        noise = self.inference_step(batch, time_steps)

        scores = self.diffusion_process.score_function(
            noise, batch[properties.idx_m], time_steps[batch[properties.idx_m]]
        )

        scores[time_steps == 0, :] = 0

        # perform one reverse step
        return scores


class ReverseODEHeun(ReverseODEEuler):
    """
    Implements a 'Reverse ODE' using Heun's method.
    """

    def __init__(self, *args, **kwargs):
        # Pass all arguments to the `DDPM` constructor
        super().__init__(*args, **kwargs)

    def get_increment(
        self, batch: Dict[str, torch.Tensor], time_steps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # broadcast the time step to atoms-level
        time_steps = time_steps[batch[properties.idx_m]]

        x_t = batch[properties.R]

        scores_0 = super().get_increment(batch, time_steps)

        # update the time for every molecule that is not at t_0
        time_steps_1 = time_steps - 1
        time_steps_1[time_steps_1 < 0] = 0
        scores_1 = super().get_increment(batch, time_steps_1)

        # restore positions
        batch[properties.R] = x_t

        # take average of scores/gradients
        increment = 0.5 * (scores_0 + scores_1)

        # for every molecule at t<=1 we just take the first score/gradient
        increment[time_steps <= 1, :] = scores_0[time_steps <= 1, :]

        # update the batch with the averaged scores
        return increment