from typing import Dict, Optional
import torch
from torch import nn
from schnetpack.model import AtomisticModel

__all__ = ["ConsistencyParametrization"]


class ConsistencyParametrization(AtomisticModel):
    """
    Consistency model parameterization which enforces the boundary condition proposed by Song et al 2021.
    Uses the boundary condition f(x, epsilon) = x.
    """

    def __init__(
        self,
        source_model: AtomisticModel,
        input_key: str,
        output_key: str,
        time_key: str,
        epsilon: float = 0,
        sigma_data: Optional[float] = 0.5,
        **kwargs,
    ):
        """
        Args:
            model: model to wrap.
            sigma_data: constant for the interpolation function. When None use non-smooth/conditional parametrization.
            epsilon: small constant to avoid division by zero.
            time_key: key to use for interpolation functions.
            output_key: key to interpolate.
        """
        super().__init__(kwargs)

        self.source_model = source_model
        self.epsilon = epsilon
        self.sigma_data = sigma_data
        if sigma_data is not None:
            self.sigma_data_sq = sigma_data**2
        self.time_key = time_key
        self.input_key = input_key
        self.output_key = output_key

        self.collect_derivatives()

    def c_skip(self, t):
        if self.sigma_data is None:
            return t == 0
        return self.sigma_data_sq / (
            torch.square(t - self.epsilon) + self.sigma_data_sq
        )

    def c_out(self, t):
        if self.sigma_data is None:
            return t != 0
        return (
            self.sigma_data
            * (t - self.epsilon)
            / torch.sqrt(self.sigma_data_sq + torch.square(t))
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # calculate interpolation coefficients
        t = inputs[self.time_key]

        # perform forward pass on wrapped model
        model_output = self.source_model(inputs)

        # interpolate between model output and input
        c_out = self.c_out(t).unsqueeze(-1)
        c_skip = self.c_skip(t).unsqueeze(-1)
        model_output[self.output_key] = model_output[self.output_key] * c_out + inputs[self.input_key] * c_skip
        
        inputs.update(model_output)

        return inputs
