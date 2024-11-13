from typing import Dict
import torch
from torch import nn

__all__ = ["ConsistencyParameterization"]
class ConsistencyParameterization(nn.Module):
    """
    Consistency model parameterization which enforces the boundary condition proposed by Song et al 2021.
    Uses the boundary condition f(x, epsilon) = x.
    """

    def __init__(
        self,
        model: nn.Module,
        input_key: str,
        output_key: str,
        time_key: str,
        epsilon: float = 1e-5,
        sigma_data: float = 0.5,
    ):
        """
        Args:
            model: model to wrap.
            sigma_data: constant for the interpolation function.
            epsilon: small constant to avoid division by zero.
            time_key: key to use for interpolation functions.
            output_key: key to interpolate.
        """
        super().__init__()

        self.model = model
        self.epsilon = epsilon
        self.sigma_data = sigma_data
        self.sigma_data_sq = sigma_data ** 2
        self.time_key = time_key
        self.input_key = input_key
        self.output_key = output_key

    def c_skip(self, t):
        return self.sigma_data_sq / (torch.square(t-self.epsilon) + self.sigma_data_sq)
    
    def c_out(self, t):
        return self.sigma_data * (t - self.epsilon) / torch.sqrt(self.sigma_data_sq + torch.square(t))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # calculate interpolation coefficients
        t = inputs[self.time_key]
        c_out = self.c_out(t)
        c_skip = self.c_skip(t)

        # perform forward pass on wrapped model
        model_out = self.model(inputs)

        # interpolate between model output and input
        interpolation = model_out[self.output_key] * c_out + inputs[self.input_key] * c_skip

        # update output
        model_out[self.output_key] = interpolation

        return model_out