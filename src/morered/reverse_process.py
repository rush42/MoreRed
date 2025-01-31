from abc import abstractmethod
import logging
from typing import Dict, List, Tuple, Union, Optional

import torch
from schnetpack import properties
from torch import nn
from tqdm import tqdm

from morered.processes import DiffusionProcess
from morered.processes.functional import _check_shapes, sample_isotropic_Gaussian
from morered.utils import compute_neighbors, scatter_mean

__all__ = ["ReverseProcess", "ReverseODE", "ReverseODEHeun", "ReverseUnbiasedEstimator"]


class ReverseProcess(nn.Module):
    """
    Abstract Class for reverse processes.
    """
    def __init__(self, diffusion_process: DiffusionProcess, time_key: str = "t"):
        super().__init__()
        self.diffusion_process = diffusion_process
        self.time_key = time_key
        pass

    @abstractmethod
    def forward(
        self, inputs: Dict[str, torch.Tensor], t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class ReverseODE(ReverseProcess):
    """
    Reverse process defined through an ODE.
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
        super().__init__(diffusion_process=diffusion_process, time_key=time_key)
        self.denoiser = denoiser
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

    def get_drift(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sqrt_alphas = self.diffusion_process.noise_schedule(t, keys=["sqrt_alpha"])[
            "sqrt_alpha"
        ]

        return (2 - sqrt_alphas).unsqueeze(-1) * x_t

    def get_diffusion(
        self, noise: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        scores = self.diffusion_process.score_function(noise, t)

        scores[t == 0, :] = 0
        betas = self.diffusion_process.noise_schedule(t, keys=["beta"])["beta"]

        return 1 / 2 * betas.unsqueeze(-1) * scores

    @torch.no_grad()
    def get_time_steps(
        self, inputs: Dict[str, torch.Tensor], iter: int
    ) -> torch.Tensor:
        # current reverse time step
        time_steps = torch.full_like(
            inputs[properties.idx_m],
            fill_value=iter,
            dtype=torch.int64,
            device=self.device,
        )
        return time_steps

    @torch.no_grad()
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
    def forward(
        self, inputs: Dict[str, torch.Tensor], t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get x_t_next.

        Args:
            inputs: input data for noise prediction.
            t: the unnormalized time step for each atom.
        """

        # append the normalized time step to the model input
        if t is None:
            t = self.diffusion_process.unnormalize_time(inputs[self.time_key])
        else:
            inputs[self.time_key] = self.diffusion_process.normalize_time(t)

        # cast input to float for the denoiser
        for key, val in inputs.items():
            if val.dtype == torch.float64:
                inputs[key] = val.float()

        # forward pass through the denoiser
        model_out = self.denoiser(inputs)
        # fetch the noise prediction
        noise_pred = model_out[self.noise_pred_key].detach()

        x_t = inputs[properties.R]

        x_t_next = self.get_drift(x_t, t) - self.get_diffusion(noise_pred, t)

        return x_t_next

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

        batch = self.prepare_batch(inputs)

        hist = []

        # simulate the reverse process
        for i in tqdm(range(t - 1, 0, -1)):
            # update the neighbors list if required
            if self.recompute_neighbors:
                batch = compute_neighbors(batch, cutoff=self.cutoff, device=self.device)

            # current reverse time step
            time_steps = self.get_time_steps(inputs, i)

            x_t_next = self(batch, time_steps)

            # save history if required. Must be done before the reverse step.
            if self.save_progress and (i % self.progress_stride == 0):
                hist.append(
                    {
                        properties.R: batch[properties.R].cpu().float().clone(),
                        self.time_key: time_steps.cpu(),
                    }
                )

            # update the state
            batch[properties.R] = x_t_next

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


class ReverseODEHeun(ReverseODE):
    @torch.no_grad()
    def forward(
        self, inputs: Dict[str, torch.Tensor], t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get x_t_next.

        Args:
            inputs: input data for noise prediction.
            t: the unnormalized time step for each atom.
        """

        # either append the normalized time step to the model input or get t by unnormalizing form inputs[self.time_key]
        if t is None:
            t = self.diffusion_process.unnormalize_time(inputs[self.time_key])
        else:
            inputs[self.time_key] = self.diffusion_process.normalize_time(t)

        x_t_intermediate = super()(inputs, t)
        t_intermediate = t - 1

        x_t = (super()(x_t_intermediate, t_intermediate) + x_t_intermediate) / 2
        x_t[t_intermediate == 0] = x_t_intermediate[t_intermediate == 0]

        return x_t


class ReverseUnbiasedEstimator(ReverseProcess):
    """
    The unbiased score estimator proposed by Song et. al. 2022.
    This is meant to be used in conjunction with the above `Diffuse` transfrom.
    """

    def __init__(
        self,
        diffuse_property: str,
        diffusion_process: DiffusionProcess,
        time_key: str = "t",
    ):
        """
        Args:
            diffuse_property: molecular property to diffuse.
            diffusion_process: the forward diffusion process to use.
            output_key: key to store the diffused property.
                        if None, the diffuse_property key is used.
            time_key: key to save the normalized diffusion time step.
        """
        super().__init__(diffusion_process=diffusion_process, time_key=time_key)
        self.diffuse_property = diffuse_property
        self.noise_key = self.diffusion_process.noise_key

        # Sanity check
        if (
            not self.diffusion_process.invariant
            and self.diffuse_property == properties.R
        ):
            logging.error(
                "Diffusing atom positions R without invariant constraint"
                "(invariant=False) might lead to unexpected results."
            )

    @torch.no_grad()
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Define the forward diffusion transformation.

        Args:
            inputs: dictionary of input tensors as in SchNetPack.
        """
        dtype = inputs[f"original_{self.diffuse_property}"].dtype

        # get x_0, t and noise from inputs
        x_0 = inputs[f"original_{self.diffuse_property}"].to(dtype)
        noise = inputs[self.noise_key]

        # take one step towards the data in normalized space
        t = self.diffusion_process.unnormalize_time(inputs[self.time_key])
        t_next = t - 1

        # query noise parameters.
        x_0, t_next = _check_shapes(x_0, t_next)
        mean, std = self.diffusion_process.perturbation_kernel(x_0, t_next)

        # sample by Gaussian diffusion.
        x_t_next, _ = sample_isotropic_Gaussian(
            mean, std, invariant=self.diffusion_process.invariant, idx_m=None, noise=noise
        )

        return x_t_next
