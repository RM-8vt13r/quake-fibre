"""
A class describing a physical event which imposes a perturbation on an optical fibre (see perturbation.py and fibre_marcuse.py)
"""
from abc import ABC, abstractmethod
import logging

import numpy as np

from .perturbation import Perturbation
from .path import Path
from .signal import Signal

logger = logging.getLogger()

class PerturbationEvent(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, path: Path, *args, **kwargs):
        return self.request_perturbations(path, *args, **kwargs)

    def request_fibre_strains(self, path: Path, *args, **kwargs) -> Signal:
        """
        Request the material strain on each path section.

        Inputs:
        - path [Path]: Fibre path with C edges
        
        Outputs:
        - [Signal] signal containing fibre strain, shape [C, T, 1].
        """
        return None

    def request_fibre_twists(self, path: Path, *args, **kwargs) -> Signal:
        """
        Request the material twists on each path section.

        Inputs:
        - path [Path]: Fibre path with C edges
        
        Outputs:
        - [Signal] signal containing fibre twists, shape [C, T, 1].
        """
        return None

    def request_perturbations(self, path: Path, *args, **kwargs) -> tuple:
        """
        Obtain the perturbations over time along a given path.

        Inputs:
        - path [Path]: the path along which to obtain perturbations.
        - args: any positional arguments to a subclassing event's implementation of this function
        - kwargs: any keyword-indexed arguments to a subclassing event's implementation of this function

        Outputs:
        - [tuple] the Perturbations along the path over time, resulting from this event. Shape [K, T], where K is the number of path segments and T the number of time samples.
        """
        strains = self.request_fibre_strains(path, *args, **kwargs)
        twists  = self.request_fibre_twists(path, *args, **kwargs)

        assert strains is None or twists is None or (strains.shape == twists.shape and strains.sample_rate == twists.sample_rate), f"Strains and twists must have the same shape and sample rate"

        perturbation = Perturbation(
                strains     = strains.samples_time[:, :, 0] if strains is not None else None,
                twists      = twists.samples_time[:, :, 0] if twists is not None else None,
                sample_rate = strains.sample_rate,
                domain      = strains.domain
            )

        logger.debug("Returning perturbation")
        return perturbation