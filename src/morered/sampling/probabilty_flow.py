from typing import Dict, List, Tuple, Optional, Callable

import torch
from schnetpack import properties
from tqdm import tqdm

from morered.sampling.ddpm import DDPM
from morered.utils import compute_neighbors, scatter_mean

__all__ = ["ProbabilityFlow"]


class ProbabilityFlow(DDPM):
    """
    Implements a 'Probabilty Flow' using the euler method.
    Subclasses the 'DDPM' class.
    """

    def __init__(self, *args, **kwargs):
        # Pass all arguments to the `DDPM` constructor
        super().__init__(*args, **kwargs)

    @torch.no_grad()
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

        hist = []

        # simulate the reverse process
        for i in tqdm(range(t - 1, -1, -1)):
            # update the neighbors list if required
            if self.recompute_neighbors:
                batch = compute_neighbors(batch, cutoff=self.cutoff, device=self.device)

            # get the time steps and noise predictions from the denoiser
            time_steps, noise = self.inference_step(batch, i)

            # save history if required. Must be done before the reverse step.
            if self.save_progress and (i % self.progress_stride == 0):
                hist.append(
                    {
                        properties.R: batch[properties.R].cpu().float().clone(),
                        self.time_key: time_steps.cpu(),
                    }
                )

            scores = self.diffusion_process.score_function(
                noise, batch[properties.idx_m], time_steps[inputs[properties.idx_m]]
            )
            # perform one reverse step
            batch[properties.R] -= scores

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