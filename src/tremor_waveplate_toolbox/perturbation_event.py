"""
A class describing a physical event which imposes a perturbation on an optical fibre (see perturbation.py and fibre_marcuse.py)
"""
from abc import ABC, abstractmethod

import numpy as np

from .perturbation import Perturbation
from .path import Path
from .signal import Signal

class PerturbationEvent(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, path: Path, step_length: float = None, *args, **kwargs):
        return self.get_perturbations(path, step_length, *args, **kwargs)

    def request_fibre_strains(self, path: Path, step_length: float = None, *args, **kwargs) -> Signal:
        """
        Request the material strain on each path section.

        Inputs:
        - path [Path]: Fibre path with C edges
        - step_length [float]: if not None, request strains at I points along path spaced step_length apart in km. Then, interpolate the results back to every edge centre along path to calculate strains.
        
        Outputs:
        - [Signal] signal containing fibre strain, shape [C, T, 1].
        """
        return (None,)

    def request_fibre_twists(self, path: Path, step_length: float = None, *args, **kwargs) -> Signal:
        """
        Request the material twists on each path section.

        Inputs:
        - path [Path]: Fibre path with C edges
        - step_length [float]: if not None, request twists at I points along path spaced step_length apart in km. Then, interpolate the results back to every edge centre along path to calculate twists.
        
        Outputs:
        - [Signal] signal containing fibre twists, shape [C, T, 1].
        """
        return (None,)

    # @abstractmethod
    def get_perturbations(self, path: Path, step_length: float = None, filter_frequencies: np.ndarray = None, filter_taps: np.ndarray = None, *args, **kwargs) -> tuple:
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
        strains, = self.request_fibre_strains(path = path, step_length = step_length, *args, **kwargs)
        twists,  = self.request_fibre_twists(path = path, step_length = step_length, *args, **kwargs)

        assert strains is None or twists is None or (strains.shape == twists.shape and strains.sample_rate == twists.sample_rate), f"Strains and twists must have the same shape and sample rate"

        if filter_frequencies is not None or filter_taps is not None:
            assert filter_frequencies is not None and filter_taps is not None, "Either filter_frequencies and filter_taps must both be None, or both be defined"
            assert len(filter_frequencies.shape) == 1, f"filter_frequencies must have only one dimension, but had shape {filter_frequencies.shape}"
            assert filter_frequencies.shape == filter_taps.shape, f"filter_frequencies and filter_taps should have the same shape, but had shapes {filter_frequencies.shape} and {filter_taps.shape}"
            
            strains.samples_frequency *= np.interp(strains.frequency % strains.bandwidth, filter_frequencies, filter_taps)[None, :, None]
            twists.samples_frequency  *= np.interp(twists.frequency % twists.bandwidth, filter_frequencies, filter_taps)[None, :, None]

        perturbation = Perturbation(
                strains     = strains.samples_time[:, :, 0] if strains is not None else None,
                twists      = twists.samples_time[:, :, 0] if twists is not None else None,
                sample_rate = strains.sample_rate,
                domain      = strains.domain
            )

        logger.debug("Returning perturbation")
        return perturbation