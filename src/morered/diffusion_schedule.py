import math
from torch import nn

# global variable to access from different threads
step = 0


class DiffusionSchedule(nn.Module):
    """
    Base class for noise schedules. To be used together with Markovian processes,
    i.e. inheriting from ``MarkovianDiffusion``.
    """

    def __init__(
        self,
        mu_0: float = 0.9,
        T_0: int = 2,
        T_1: int = 64,
        K: int = 1e5,
    ):
        """
        Args:
            mu_0: the initial 
        """
        super().__init__()
        self.mu_0 = mu_0
        self.T_0 = T_0
        self.T_1 = T_1
        self.K = K

    def get_T(self):
        global step
        return math.ceil(
            math.sqrt(
                step / self.K((self.T_1 + 1) ** 2 - self.T_0**2) + self.T_0**2 - 1
            )
        )
    
    def get_ema_decay(self):
        math.exp(self.T_0 * math.log(self.mu_0) / self.get_T())

    def take_step(self):
        global step
        step += 1
