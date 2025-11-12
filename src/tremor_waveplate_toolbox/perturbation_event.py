"""
A class describing a physical event which imposes a perturbation on an optical fibre (see perturbation.py and fibre_marcuse.py)
"""
from abc import ABC, abstractmethod

from .perturbation import Perturbation
from .path import Path

class PerturbationEvent(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, path: Path, *args, **kwargs):
        return self.get_perturbation(path, *args, **kwargs)

    @abstractmethod
    def get_perturbation(self, path: Path, *args, **kwargs) -> Perturbation:
        """
        Obtain the perturbations over time along a given path.

        Inputs:
        - path [Path]: the path along which to obtain perturbations.
        - args: any positional arguments to a subclassing event's implementation of this function
        - kwargs: any keyword-indexed arguments to a subclassing event's implementation of this function

        Outputs:
        - [Perturbation] the Perturbation along the path over time, resulting from this event. Shape [K, T], where K is the number of path segments and T the number of time samples.
        """
        pass