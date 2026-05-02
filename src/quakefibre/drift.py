"""
A perturbation event that causes slow random SOP drift along the fibre at the carrier frequency.
Note: the current implementation is a simplified placeholder, and by no means theoretically accurate!
"""
from typing import override
import logging

import numpy as np

from .perturbation_event import PerturbationEvent
from .perturbation import Perturbation
from .path import Path

logger = logging.getLogger()

class Drift(PerturbationEvent):
    def __init__(self):
        logger.error("Drift is an ad-hoc, untested implementation and cannot be assumed to be correct")
        super().__init__()

    def _get_drift(path: Path, drift_scalar: float, sample_rate: float, sample_count: int):
        drift_standard_deviations = drift_scalar * path.lengths / sample_rate
        drift_steps = drift_standard_deviations[:, None] * np.random.default_rng().normal(size = (path.edge_count, sample_count))
        return np.cumsum(drift_steps, axis = 1)

    @override
    def request_fibre_strains(self, path: Path, drift_scalar: float, sample_rate: float, sample_count: int) -> Perturbation:
        """
        Request strains that change by slow polarisation drift everywhere along the fibre.
        
        Inputs:
        - path [Path]: the path of the optical fibre
        - drift_scalar [float]: the state of polarisation drifts slowly with a standard deviation of drift_scalar * section length / sample_rate; the unit of drift_scalar is rad / m / s
        - sample_rate [float]: the inverse duration in Hz between two subsequent time indices
        - sample_count [int]: how long the signal should be

        Outputs:
        - [Signal] the resulting slowly drifting fibre strains
        """
        return self._get_drift(path, drift_scalar, sample_rate, sample_count)
        
    @override
    def request_fibre_twists(self, path: Path, drift_scalar: float, sample_rate: float, sample_count: int) -> Perturbation:
        """
        Request twists that change by slow polarisation drift everywhere along the fibre.
        
        Inputs:
        - path [Path]: the path of the optical fibre
        - drift_scalar [float]: the state of polarisation drifts slowly with a standard deviation of drift_scalar * section length / sample_rate; the unit of drift_scalar is rad / m / s
        - sample_rate [float]: the inverse duration in Hz between two subsequent time indices
        - sample_count [int]: how long the signal should be

        Outputs:
        - [Signal] the resulting slowly drifting fibre twists
        """
        return self._get_drift(path, drift_scalar, sample_rate, sample_count)
