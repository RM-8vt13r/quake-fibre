"""
A perturbation event that randomly scrambles the state of polarisation at the carrier frequency in discrete places along the fibre.
Note: the current implementation is a simplified placeholder, and by no means theoretically accurate!
"""
from typing import override

import numpy as np

from .perturbation_event import PerturbationEvent
from .perturbation import Perturbation
from .path import Path

class Scramblers(PerturbationEvent):
    @override
    def get_perturbation(self, path: Path, vertex_indices: np.ndarray, time_indices: np.ndarray, sample_rate: float, sample_count: int) -> Perturbation:
        """
        Create a perturbation that randomly scrambles the state of polarisation at several locations and times along the fibre.
        
        Inputs:
        - path [Path]: the path of the optical fibre
        - vertex_indices [np.ndarray]: vertices along the path where the state of polarisation must be scrambled, shape [P,]
        - time_indices [np.ndarray]: the time samples where the state of polarisation must be scrambled, shape [P,]
        - sample_rate [float]: the inverse duration in Hz between two subsequent time indices
        - sample_count [int]: how long the signal should be

        Outputs:
        - [Perturbation] the resulting scrambler perturbation
        """
        vertex_indices = np.array(vertex_indices)
        time_indices = np.array(time_indices)

        assert len(vertex_indices.shape) == 1, f"vertex_indices must have shape [P,], but had shape {vertex_indices.shape}"
        assert vertex_indices.shape == time_indices.shape, f"vertex_indices and time_indices must have the same shapes [P,], but had shapes {vertex_indices.shape} and {time_indices.shape}"
        assert np.all((vertex_indices >= 0) & (vertex_indices < path.edge_count)), f"All vertex_indices must index an existing path edge (>= 0, < path.edge_count), but not all did"
        assert np.all((time_indices >= 0) & (time_indices < sample_count)), f"All time_indices must index an time sample (>= 0, < sample_count), but not all did"

        # Build separate perturbation signals
        sorter = np.argsort(time_indices)
        time_indices = time_indices[sorter]
        vertex_indices = vertex_indices[sorter]

        birefringence_adders = np.zeros(shape = (path.edge_count, sample_count), dtype = float)
        major_angles_adders = np.zeros_like(birefringence_adders)

        for vertex_index, time_index in zip(vertex_indices, time_indices):
            birefringence_adders[vertex_index, time_index:] += np.random.default_rng().uniform(low = -np.pi / 2, high = np.pi / 2)
            major_angles_adders[vertex_index, time_index:] += np.random.default_rng().uniform(low = -np.pi, high = np.pi)
            
        # Build perturbation
        perturbation = Perturbation(
            birefringence_adders = birefringence_adders,
            major_angles_adders = major_angles_adders,
            sample_rate = sample_rate
        )

        return perturbation