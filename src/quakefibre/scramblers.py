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
    def __init__(self):
        logger.error("Scramblers is an ad-hoc, untested implementation and cannot be assumed to be correct")
        super().__init__()

    def _prepare_inputs(self, path: Path, vertex_indices: np.ndarray, time_indices: np.ndarray):
        vertex_indices = np.array(vertex_indices)
        time_indices = np.array(time_indices)

        assert len(vertex_indices.shape) == 1, f"vertex_indices must have shape [P,], but had shape {vertex_indices.shape}"
        assert vertex_indices.shape == time_indices.shape, f"vertex_indices and time_indices must have the same shapes [P,], but had shapes {vertex_indices.shape} and {time_indices.shape}"
        assert np.all((vertex_indices >= 0) & (vertex_indices < path.edge_count)), f"All vertex_indices must index an existing path edge (>= 0, < path.edge_count), but not all did"
        assert np.all((time_indices >= 0) & (time_indices < sample_count)), f"All time_indices must index an time sample (>= 0, < sample_count), but not all did"

        return vertex_indices, time_indices

    def _get_scrambles(self, path: Path, vertex_indices: np.ndarray, time_indices: np.ndarray, sample_count: int, low: float, high: float):
        # Build perturbation signal
        sorter = np.argsort(time_indices)
        time_indices = time_indices[sorter]
        vertex_indices = vertex_indices[sorter]

        scrambles = np.zeros(shape = (path.edge_count, sample_count), dtype = float)

        for vertex_index, time_index in zip(vertex_indices, time_indices):
            scrambles[vertex_index, time_index:] += np.random.default_rng().uniform(low = low, high = high)

        return scrambles

    @override
    def request_fibre_strains(self, path: Path, vertex_indices: np.ndarray, time_indices: np.ndarray, sample_count: int) -> Perturbation:
        """
        Create random uniform fibre strain at several locations and times along the fibre.
        
        Inputs:
        - path [Path]: the path of the optical fibre
        - vertex_indices [np.ndarray]: vertices along the path where the state of polarisation must be scrambled, shape [P,]
        - time_indices [np.ndarray]: the time samples where the state of polarisation must be scrambled, shape [P,]
        - sample_count [int]: how long the signal should be

        Outputs:
        - [Signal] the resulting scrambled fibre strains
        """
        vertex_indices, time_indices = self._prepare_inputs(path, vertex_indices, time_indices)
        return self._get_scrambles(path, vertex_indices, time_indices, sample_count, low = -np.pi / 2, high = np.pi / 2)

    @override
    def request_fibre_twists(self, path: Path, vertex_indices: np.ndarray, time_indices: np.ndarray, sample_count: int) -> Perturbation:
        """
        Create random uniform fibre twist at several locations and times along the fibre.
        
        Inputs:
        - path [Path]: the path of the optical fibre
        - vertex_indices [np.ndarray]: vertices along the path where the state of polarisation must be scrambled, shape [P,]
        - time_indices [np.ndarray]: the time samples where the state of polarisation must be scrambled, shape [P,]
        - sample_count [int]: how long the signal should be

        Outputs:
        - [Signal] the resulting scrambled fibre twists
        """
        vertex_indices, time_indices = self._prepare_inputs(path, vertex_indices, time_indices)
        return self._get_scrambles(path, vertex_indices, time_indices, sample_count, low = -np.pi, high = np.pi)