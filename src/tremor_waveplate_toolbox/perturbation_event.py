"""
A class describing a physical event which imposes a perturbation on an optical fibre (see perturbation.py and fibre_marcuse.py)
"""
from abc import ABC, abstractmethod

from .perturbation import Perturbation
from .path import Path

class PerturbationEvent(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, path: Path, step_length: float = None, *args, **kwargs):
        return self.get_perturbations(path, step_length, *args, **kwargs)

    @abstractmethod
    def get_perturbations(self, path: Path, step_length: float = None, *args, **kwargs) -> tuple:
        """
        Obtain the perturbations over time along a given path.

        Inputs:
        - path [Path]: the path along which to obtain perturbations.
        - step_length [float]: if not None, generate perturbations at points along path spaced step_length apart in km. Then, interpolate the results back to every vertex along path.
        - args: any positional arguments to a subclassing event's implementation of this function
        - kwargs: any keyword-indexed arguments to a subclassing event's implementation of this function

        Outputs:
        - [tuple] the Perturbations along the path over time, resulting from this event. Shape [K, T], where K is the number of path segments and T the number of time samples.
        """
        pass