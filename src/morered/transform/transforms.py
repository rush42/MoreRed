import logging
from typing import Dict, Optional, Union

import schnetpack.transform as trn
import torch
from torch import nn
from schnetpack import properties

from morered.diffusion_schedule import DiffusionSchedule
from morered.processes.base import DiffusionProcess
from morered.processes.functional import sample_isotropic_Gaussian
from morered.utils import batch_center_systems

__all__ = [
    "AllToAllNeighborList",
    "BatchSubtractCenterOfMass",
    "Diffuse",
]

class AllToAllNeighborList(trn.NeighborListTransform):
    """
    Calculate a full neighbor list for all atoms in the system.
    Faster than other methods and useful for small systems.
    """

    def __init__(self):
        # pass dummy large cutoff as all neighbors are connceted
        super().__init__(cutoff=1e8)

    def _build_neighbor_list(self, Z, positions, cell, pbc, cutoff):
        n_atoms = Z.shape[0]
        idx_i = torch.arange(n_atoms).repeat_interleave(n_atoms)
        idx_j = torch.arange(n_atoms).repeat(n_atoms)

        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        offset = torch.zeros(n_atoms * (n_atoms - 1), 3, dtype=positions.dtype)
        return idx_i, idx_j, offset


class BatchSubtractCenterOfMass(trn.Transform):
    """
    subsctract center of mass from input systems batchwise.
    """

    is_preprocessor: bool = False
    is_postprocessor: bool = True
    force_apply: bool = True

    def __init__(
        self,
        name: str = "eps_pred",
        dim: int = 3,
    ):
        """
        Args:
            name: name of the property to be centered.
            dim: number of dimensions of the property to be centered.
        """
        super().__init__()
        self.name = name
        self.dim = dim

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        forward pass of the transform.

        Args:
            inputs: dictionary of input tensors.
        """
        # check shapes
        if inputs[self.name].shape[1] < self.dim:
            raise ValueError(
                f"Property {self.name} has less than {self.dim} dimensions. "
                f"Cannot subtract center of mass."
            )

        # center batchwise
        if inputs[self.name].shape[-1] == self.dim:
            inputs[self.name] = batch_center_systems(
                inputs[self.name], inputs[properties.idx_m], inputs[properties.n_atoms]
            )
        # use the first dimensions if the property has more than 'dim' dimensions.
        else:
            x = inputs[self.name][:, : self.dim]
            h = inputs[self.name][:, self.dim :]
            x_cent = batch_center_systems(
                x, inputs[properties.idx_m], inputs[properties.n_atoms]
            )
            inputs[self.name] = torch.cat((x_cent, h), dim=-1).to(
                device=inputs[self.name].device
            )

        return inputs


class Diffuse(trn.Transform):
    """
    Wrapper class for diffusion process of molecular properties.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        diffuse_property: str,
        diffusion_process: DiffusionProcess,
        output_key: Optional[str] = None,
        include_t_0: bool = True,
        time_key: str = "t",
        t_1_bonus: int = 0.0,
        diffusion_range: float = 1.0,
        diffusion_schedule: Optional[DiffusionSchedule] = None,
    ):
        """
        Args:
            diffuse_property: molecular property to diffuse.
            diffusion_process: the forward diffusion process to use.
            output_key: key to store the diffused property.
                        if None, the diffuse_property key is used.
            time_key: key to save the normalized diffusion time step.
            include_t_0: whether to produce undiffused samples
            t_1_bonus: probability to favor t_1
            diffusion_range: a fraction which determines the maximum diffusion
        """
        super().__init__()
        self.diffuse_property = diffuse_property
        self.diffusion_process = diffusion_process
        self.output_key = output_key
        self.time_key = time_key
        self.include_t_0 = include_t_0
        self.t_1_bonus = t_1_bonus
        if diffusion_range < 0 or diffusion_range > 1:
            raise "diffusion_range needs to be between 0 and 1"
        self.diffusion_range = diffusion_range
        self.diffusion_schedule = diffusion_schedule
        # Sanity check
        if (
            not self.diffusion_process.invariant
            and self.diffuse_property == properties.R
        ):
            logging.error(
                "Diffusing atom positions R without invariant constraint"
                "(invariant=False) might lead to unexpected results."
            )

    def sample_t(self, device):
        t_low = 0 if self.include_t_0 else 1

        t_high = round((self.diffusion_process.get_T() * self.diffusion_range))

        if self.diffusion_schedule is not None:
            t_high = self.diffusion_schedule.get_range()

        t = torch.randint(
            t_low,
            t_high,
            size=(1,),
            dtype=torch.long,
            device=device,
        )

        if self.t_1_bonus != 0:
            if (torch.rand(size=(1,)) < self.t_1_bonus).all():
                t = torch.tensor(1)

        return t

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Define the forward diffusion transformation.

        Args:
            inputs: dictionary of input tensors as in SchNetPack.
        """
        x_0 = inputs[self.diffuse_property]

        # save the original value.
        outputs = {
            f"original_{self.diffuse_property}": x_0,
        }

        # sample 't' for the input molecule.
        t = self.sample_t(device=x_0.device)

        # diffuse the property.
        tmp = self.diffusion_process.diffuse(
            x_0,
            idx_m=None,
            t=t,
            return_dict=True,
            output_key=self.output_key or self.diffuse_property,
        )
        outputs.update(tmp)

        # broadcast the diffusion time step to all atoms.
        outputs[self.time_key] = t.repeat(inputs[properties.n_atoms])

        # normalize the time step to [0,1].
        outputs[self.time_key] = self.diffusion_process.normalize_time(
            outputs[self.time_key]
        )

        # update the returned inputs.
        inputs.update(outputs)

        return inputs
