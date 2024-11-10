from abc import abstractmethod
from typing import Dict, List, Tuple, Optional, Callable

import torch
from schnetpack import properties
from tqdm import tqdm

from morered.sampling.ddpm import DDPM
from morered.utils import compute_neighbors

__all__ = ["ProbabilityFlowEuler", "ProbabilityFlowHeun", "ProbabilityFlow"]


class ProbabilityFlow(DDPM):
    """
    Abstract base class to define a 'Probabilty Flow' using different ODE solvers.
    """

    def __init__(self, *args, **kwargs):
        # Pass all arguments to the `DDPM` constructor
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_increment(self, batch, i):
        """
        Get the increment for the reverse ODE process.
        Args:
            batch: dict with input data in the SchNetPack form
            i: the time step of the reverse process.
        """
        raise NotImplementedError

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

            # get the time steps and noise predictions from the denoiser
            time_steps, increment = self.get_increment(batch, i)

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


class ProbabilityFlowEuler(ProbabilityFlow):
    """
    Implements a 'Probabilty Flow' using Euler's method.
    Subclasses the 'DDPM' class.
    """

    def __init__(self, *args, **kwargs):
        # Pass all arguments to the `DDPM` constructor
        super().__init__(*args, **kwargs)

    def get_increment(self, batch, i):
        # get the time steps and noise predictions from the denoiser
        time_steps, noise = self.inference_step(batch, i)

        scores = self.diffusion_process.score_function(
            noise, batch[properties.idx_m], time_steps[batch[properties.idx_m]]
        )
        # perform one reverse step
        return time_steps, scores


class ProbabilityFlowHeun(ProbabilityFlow):
    """
    Implements a 'Probabilty Flow' using Heun's method.
    """

    def __init__(self, *args, **kwargs):
        # Pass all arguments to the `DDPM` constructor
        super().__init__(*args, **kwargs)

    def get_increment(self, batch, i):
        # get the time steps and noise predictions from the denoiser
        time_steps, noise = self.inference_step(batch, i)

        scores = self.diffusion_process.score_function(
            noise, batch[properties.idx_m], time_steps[batch[properties.idx_m]]
        )

        if i == 0:
            return time_steps, scores

        # perform one reverse step to get the intermediate state
        intermediate_state = batch.copy()
        intermediate_state[properties.R] = batch[properties.R] - scores

        # get the time steps and noise predictions  for the intermediate state
        intermediate_time_steps, intermediate_noise = self.inference_step(intermediate_state, i - 1)

        intermediate_scores = self.diffusion_process.score_function(
            intermediate_noise,
            batch[properties.idx_m],
            intermediate_time_steps[batch[properties.idx_m]],
        )

        # update the batch with the averaged scores
        return time_steps, 0.5 * (scores + intermediate_scores)
