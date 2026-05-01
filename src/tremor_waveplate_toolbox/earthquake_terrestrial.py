"""
A class that simulates earthquakes on land along a path on the earth using Syngine.
"""
import logging
from typing import override
from configparser import ConfigParser

import numpy as np
from obspy.geodetics.base import WGS84_A as earth_radius # meters

from .earthquake import Earthquake
from .signal import Signal
from .path import Path

logger = logging.getLogger()

class EarthquakeTerrestrial(Earthquake):
    @override
    def __init__(self, parameters: ConfigParser):
        logger.error("EarthquakeTerrestrial is an ad-hoc implementation and cannot be assumed to be correct")
        super().__init__(parameters)

    @override
    def _local_seismograms_build_batches(self,
                earthquake_path: Path,
                batch_size: int
            ) -> (int, list[np.ndarray], np.ndarray, np.ndarray):
        """
        Part of request_local_seismograms() that builds batches of HTTP requests to send to Syngine.
        The EarthquakeTerrestrial version puts batch coordinates on the joints between fibre sections.
        """
        return super()._local_seismograms_build_batches(earthquake_path.coordinates, batch_size)

    @override
    def _local_seismograms_postprocess(self,
                earthquake_path: Path,
                syngine_stream: list
            ):
        return super()._local_seismograms_postprocess(earthquake_path.vertex_count, syngine_stream)

    def _global_seismograms_interpolate(self, earthquake_path: Path, path: Path, displacements: np.ndarray):
        """
        Part of request_global_seismograms() that interpolates sparsely obtained seismograms to a denser path.
        """
        displacements_interpolated_flattened = np.zeros(shape = (np.prod(displacements.shape[-2:]), path.vertex_count)) # [T * D, C]
        displacements_flattened = displacements.reshape((earthquake_path.vertex_count, np.prod(displacements.shape[-2:]))).transpose() # [I, T, D] -> [I, T * D] -> [T * D, I]

        for channel_index, displacement_flattened in enumerate(displacements_flattened):
            displacements_interpolated_flattened[channel_index] = np.interp(path.positions, earthquake_path.positions, displacement_flattened)

        displacements_interpolated = displacements_interpolated_flattened.transpose().reshape((path.vertex_count, *displacements.shape[-2:])) # [T * D, C] -> [C, T * D] -> [C, T, D]

        logger.info(f"Global seismograms interpolated from {earthquake_path.vertex_count} vertices to {path.vertex_count} vertices")

        return displacements_interpolated

    def get_global_seismograms(self,
                local_seismograms: Signal,
                path: Path,
                earthquake_path: Path,
            ):
        """
        Request seismograms from Syngine at fibre section endpoints, and transform them from the local axes at each coordinate to a shared global axes system.

        Inputs:
        - local_seismograms [Signal]: signal containing all three displacement components in m, relative to local coordinates, shape [I, T, D] were D indexes longitudinal, latitudinal, and normal components in that order
        - path [Path]: Fibre path with C vertices
        - earthquake_path [Path]: interpolated version of path with I vertices spaced step_length km apart

        Outputs:
        - [Signal] signal containing all three displacement components in m, relative to global coordinates, shape [C, T, D] where D indexes global x, y, z components in that order
        """
        sin_long = np.sin(earthquake_path.longitudes)
        sin_lat  = np.sin(earthquake_path.latitudes)
        cos_long = np.cos(earthquake_path.longitudes)
        cos_lat  = np.cos(earthquake_path.latitudes)
        zeros    = np.zeros_like(sin_long)

        transformation_global_to_local = np.array([
            [-sin_long, -sin_lat * cos_long, cos_lat * cos_long],
            [ cos_long, -sin_lat * sin_long, cos_lat * sin_long],
            [ zeros   ,  cos_lat           , sin_lat           ]
        ]).transpose((2, 0, 1))

        global_seismograms = Signal(
            samples = np.einsum('cgd,ctd->ctg', transformation_global_to_local, local_seismograms.samples_time),
            sample_rate = local_seismograms.sample_rate
        )

        if earthquake_path != path:
            global_seismograms.samples_time = self._global_seismograms_interpolate(earthquake_path, path, global_seismograms.samples_time)
        
        logger.debug("Returning global seismograms")
        return global_seismograms

    def get_projected_seismograms(self,
                global_seismograms: Signal,
                path: Path,
            ):
        """
        Interpreting longitudes and latitudes as chronological path coordinates, project seismograms at each coordinate onto the neighbouring straight segments of this path.
 
        Inputs:
        - global_seismograms [Signal]: signal containing all three displacement components in m, relative to global coordinates, shape [C, T, D] where D indexes global x, y, z components in that order
        - path [Path]: Fibre path with C vertices

        Outputs:
        - [Signal] signal containing displacement in m projected onto the path, shape [C-1, T, E] where E = 2 distinguishes between section beginnings and ends.
        """
        path_coordinates_global = earth_radius * np.array([
            np.cos(path.latitudes) * np.cos(path.longitudes),
            np.cos(path.latitudes) * np.sin(path.longitudes),
            np.sin(path.latitudes)
        ]).T

        path_directions_global = path_coordinates_global[1:] - path_coordinates_global[:-1] # [C-1, D = 3]
        path_directions_global /= np.linalg.norm(path_directions_global, axis = -1)[:, None]

        projected_seismograms_samples = np.zeros(shape = (global_seismograms.shape[0] - 1, global_seismograms.shape[1], 2), dtype = float)
        projected_seismograms_samples[:, :, 0] = np.einsum('ctd,cd->ct', global_seismograms.samples_time[:-1], path_directions_global)
        projected_seismograms_samples[:, :, 1] = np.einsum('ctd,cd->ct', global_seismograms.samples_time[1:], path_directions_global)

        projected_seismograms = Signal(
            samples = projected_seismograms_samples,
            sample_rate = global_seismograms.sample_rate
        )

        logger.debug("Returning projected seismograms")
        return projected_seismograms

    @override
    def get_fibre_strains(self,
                projected_seismograms: Signal,
                path: Path,
            ):
        """
        Interpreting longitudes and latitudes as chronological path coordinates, request the longitudinal material strain on each path section by differentiating the projected seismograms in space.

        Inputs:
        - projected_seismograms [Signal]: signal containing displacement in m projected onto the path, shape [C-1, T, E] where E = 2 distinguishes between section beginnings and ends.
        - path [Path]: Fibre path with C vertices

        Outputs:
        - [Signal] signal containing strain projected onto the path, shape [C-1, T, 1].
        """
        fibre_strains = Signal(
            samples = np.diff(projected_seismograms.samples_time, axis = -1) / path.lengths[:, None, None],
            sample_rate = projected_seismograms.sample_rate
        )

        logger.debug("Returning fibre strains")
        return fibre_strains

    @override
    def request_fibre_strains(self,
            path,
            step_length,
            duration,
            batch_size,
            worker_count,
            request_delay
        ) -> Signal:
        """
        Interpreting longitudes and latitudes as chronological path coordinates, request the longitudinal material strain on each path section from Syngine.

        Inputs:
        - path [Path]: coordinates, length C
        - step_length [float]: if not None, request earthquakes at I points along path spaced step_length apart in km.
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C
        - worker_count [int]: how many Syngine requests to make at most in parallel
        - request_delay [float]: minimum delay in seconds between launching two Syngine requests

        Outputs:
        - [Signal] signal containing fibre strain, shape [C, T, 1].
        """
        earthquake_path, local_seismograms = self.request_local_seismograms(path, step_length, duration, batch_size, worker_count, request_delay)

        global_seismograms = self.get_global_seismograms(local_seismograms, path, earthquake_path)
        del earthquake_path
        del local_seismograms

        projected_seismograms = self.get_projected_seismograms(global_seismograms, path)
        del global_seismograms

        fibre_strains = self.get_fibre_strains(projected_seismograms, path)
        del projected_seismograms

        return fibre_strains