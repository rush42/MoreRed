from torch import nn

# global variable to access from different threads
epoch = 0

class DiffusionSchedule(nn.Module):
    """
    Base class for noise schedules. To be used together with Markovian processes,
    i.e. inheriting from ``MarkovianDiffusion``.
    """

    def __init__(
        self,
        T: int,
        start: int = 1,
        step_size: int = 1,
        every_n_epochs: int = 5,
    ):
        """
        Args:
            T: number of timesteps.
            include_t_0: whether to include 
        """
        super().__init__()
        self.T = T
        self.start = start
        self.every_n_epochs = every_n_epochs
        self.step_size = step_size

    def get_range(self):
        T_bar = self.start
        T_bar += self.step_size * (epoch // self.every_n_epochs)
        if T_bar < self.T:
            return T_bar
        
        return self.T
    
    
    def increase_epoch(self):
        global epoch
        epoch += 1