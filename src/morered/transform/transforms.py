import logging
from typing import Dict, Optional, Union

import schnetpack.transform as trn
import torch
from torch import nn
from schnetpack import properties

from morered.processes.base import DiffusionProcess
from morered.sampling.probabilty_flow import ProbabilityFlow
from morered.utils import batch_center_systems

__all__ = ["AllToAllNeighborList", "BatchSubtractCenterOfMass", "Diffuse", "TakeProbabilityFlowStep"]


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
        super().__init__()
        self.diffuse_property = diffuse_property
        self.diffusion_process = diffusion_process
        self.output_key = output_key
        self.time_key = time_key

        # Sanity check
        if (
            not self.diffusion_process.invariant
            and self.diffuse_property == properties.R
        ):
            logging.error(
                "Diffusing atom positions R without invariant constraint"
                "(invariant=False) might lead to unexpected results."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Define the forward diffusion transformation.

        Args:
            inputs: dictionary of input tensors as in SchNetPack.
        """
        x_0 = inputs[self.diffuse_property]
        device = x_0.device

        # save the original value.
        outputs = {
            f"original_{self.diffuse_property}": x_0,
        }

        # sample one training time step for the input molecule.
        t = torch.randint(
            0,
            self.diffusion_process.get_T(),
            size=(1,),
            dtype=torch.long,
            device=device,
        )

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


class TakeProbabilityFlowStep(trn.Transform):
    """
    Wrapper class to take a probability flow step after diffusion.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        positions_key: str,
        probability_flow: ProbabilityFlow,
        output_key: Optional[str] = None,
        time_key: str = "t",
        output_time_key: str = "t-1",
    ):
        """
        Args:
            positions_key: key to the atom positions.
            probability_flow: the probability flow to use.
            output_key: key to store the succesive postions.
                        if None, the positions_key key is used.
            time_key: key to save the normalized diffusion time step.
        """
        super().__init__()
        self.position_key = positions_key
        self.output_key = output_key or positions_key
        self.time_key = time_key
        self.probability_flow = probability_flow
        self.output_time_key = output_time_key
        # Sanity check
        if (
            not self.probability_flow.diffusion_process.invariant
            and self.position_key == properties.R
        ):
            logging.error(
                "Diffusing atom positions R without invariant constraint"
                "(invariant=False) might lead to unexpected results."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Define the probability flow transformation.

        Args:
            inputs: dictionary of input tensors as in SchNetPack.
        """
        x_t = inputs[self.position_key]

        # save the original value.
        outputs = {
            f"original_{self.position_key}": x_t,
        }

        # get the time step from the normalized time.
        Ts = self.probability_flow.diffusion_process.unnormalize_time(inputs[self.time_key])
        t = Ts[0]
        assert torch.all(Ts == t)

        if t <= 1:
            outputs[self.output_key] = inputs[f"original_{self.position_key}"] 
        else:
            # save all input keys to remove all mutations done by the denoiser
            original_keys = set(inputs.keys())

            # add idx_m to the inputs to simulate batch input.
            n_atoms = inputs[properties.n_atoms]
            assert len(n_atoms) == 1
            n_atoms = n_atoms[0].item()
            inputs[properties.idx_m] = torch.full((n_atoms,), fill_value=0)

            # take on step in the probability flow.
            _, increment = self.probability_flow.get_increment(inputs, t)
            outputs[self.output_key] = x_t - increment
            Ts -= 1

            # remove all entries that were added by the denoiser
            for key in list(inputs.keys()):
                if key not in original_keys:
                    inputs.pop(key)

        # normalize the time step to [0,1].
        outputs[self.output_time_key] = self.probability_flow.diffusion_process.normalize_time(
            Ts
        )

        # update the returned inputs.
        inputs.update(outputs)

        return inputs
