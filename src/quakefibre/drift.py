"""
A perturbation event that causes slow random SOP drift along the fibre at the carrier frequency.
Note: the current implementation is a simplified placeholder, and by no means theoretically accurate!
"""
from typing import override

import numpy as np

from .perturbation_event import PerturbationEvent
from .perturbation import Perturbation
from .path import Path

class Drift(PerturbationEvent):
    @override
    def get_perturbation(self, path: Path, drift_scalar: float, sample_rate: float, sample_count: int) -> Perturbation:
        """
        Create a perturbation that randomly imposes slow polarisation drift everywhere along the fibre.
        
        Inputs:
        - path [Path]: the path of the optical fibre
        - drift_scalar [float]: the state of polarisation drifts slowly with a standard deviation of drift_scalar * section length / sample_rate; the unit of drift_scalar is rad / m / s
        - sample_rate [float]: the inverse duration in Hz between two subsequent time indices
        - sample_count [int]: how long the signal should be

        Outputs:
        - [Perturbation] the resulting scrambler perturbation
        """
        drift_standard_deviations = drift_scalar * path.lengths / sample_rate
        birefringence_adder_steps = drift_standard_deviations[:, None] * np.random.default_rng().normal(size = (path.edge_count, sample_count))
        major_angles_adder_steps  = drift_standard_deviations[:, None] * np.random.default_rng().normal(size = (path.edge_count, sample_count))

        birefringence_adders = np.cumsum(birefringence_adder_steps, axis = 1)
        major_angles_adders  = np.cumsum(major_angles_adder_steps, axis = 1)
        
        # Build perturbation
        perturbation = Perturbation(
            birefringence_adders = birefringence_adders,
            major_angles_adders = major_angles_adders,
            sample_rate = sample_rate
        )
        return perturbation