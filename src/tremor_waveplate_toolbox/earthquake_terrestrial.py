"""
A class that simulates earthquakes on land along a path on the earth using Syngine.
"""
import logging
from typing import override

import numpy as np
from obspy.geodetics.base import WGS84_A as earth_radius # meters

from .earthquake import Earthquake
from .signal import Signal
from .path import Path

logger = logging.getLogger()

class EarthquakeTerrestrial(Earthquake):
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
    def _local_seismograms_postprocess(self, earthquake_path: Path, syngine_stream: list):
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

        logger.info(f"Seismograms interpolated from {earthquake_path.vertex_count} vertices to {path.vertex_count} vertices")

        return displacements_interpolated

    def request_global_seismograms(self,
                path: Path,
                step_length: float = None,
                duration: float = None,
                return_displacements_local: bool = False,
                batch_size: int = None,
                worker_count: int = 1,
                request_delay: float = 0.1
            ):
        """
        Request seismograms from Syngine at fibre section endpoints, and transform them from the local axes at each coordinate to a shared global axes system.

        Inputs:
        - path [Path]: Fibre path with C vertices
        - step_length [float]: if not None, request earthquakes at I points along path spaced step_length apart in km. Then, interpolate the results back to every vertex along path to calculate strains.
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - return_displacements_local [bool]: if True, return displacements in local and global coordinates. If False, return them only in global coordinates.
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C
        - worker_count [int]: how many Syngine requests to make at most in parallel
        - request_delay [float]: minimum delay in seconds between launching two Syngine requests

        Outputs:
        - If return_displacements_local: [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [I, T, D] where D indexes longitudinal, latitudinal, and normal components in that order
        - [Signal] signal containing all three displacement components in m, relative to global coordinates, shape [C, T, D] where D indexes global x, y, z components in that order
        """
        earthquake_path, displacements_local = self.request_local_seismograms(path, step_length, duration, batch_size, worker_count, request_delay)

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

        displacements_global = Signal(
            samples = np.einsum('cgd,ctd->ctg', transformation_global_to_local, displacements_local.samples_time),
            sample_rate = displacements_local.sample_rate
        )

        if step_length is not None:
            displacements_global.samples_time = self._global_seismograms_interpolate(earthquake_path, path, displacements_global.samples_time)
        
        return_list = []
        if return_displacements_local: return_list.append(displacements_local)
        return_list.append(displacements_global)

        logger.debug("Returning global seismograms")
        return return_list

    def request_projected_seismograms(self,
                path: Path,
                step_length: float = None,
                duration: float = None,
                return_displacements_local: bool = False,
                return_displacements_global: bool = False,
                batch_size: int = None,
                worker_count: int = 1,
                request_delay: float = 0.1
            ):
        """
        Interpreting longitudes and latitudes as chronological path coordinates, project seismograms at each coordinate onto the neighbouring straight segments of this path.
 
        Inputs:
        - path [Path]: Fibre path with C vertices
        - step_length [float]: if not None, request earthquakes at I points along path spaced step_length apart in km. Then, interpolate the results back to every vertex along path to calculate strains.
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - return_displacements_local [bool]: if True, return displacements in local coordinates in addition to the other return value(s).
        - return_displacements_global [bool]: if True, return displacements in global coordinates in addition to the other return value(s).
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C
        - worker_count [int]: how many Syngine requests to make at most in parallel
        - request_delay [float]: minimum delay in seconds between launching two Syngine requests

        Outputs:
        - If return_displacements_local: [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [I, T, D] with time T, and where D = 3 indexes longitudinal, latitudinal, and normal components in that order.
        - If return_displacements_global: [Signal] signal containing all three displacement components in m, relative to global coordinates, shape [C, T, D] where D = 3 indexes global x, y, and z components in that order.
        - [Signal] signal containing displacement in m projected onto the path, shape [C-1, T, E] where E = 2 distinguishes between section beginnings and ends.
        """
        results_list = self.request_global_seismograms(path, step_length, duration, return_displacements_local, batch_size, worker_count, request_delay)
        displacements_global = results_list[-1]

        path_coordinates_global = earth_radius * np.array([
            np.cos(path.latitudes) * np.cos(path.longitudes),
            np.cos(path.latitudes) * np.sin(path.longitudes),
            np.sin(path.latitudes)
        ]).T

        path_directions_global = path_coordinates_global[1:] - path_coordinates_global[:-1] # [C-1, D = 3]
        path_directions_global /= np.linalg.norm(path_directions_global, axis = -1)[:, None]

        displacements_projected_samples = np.zeros(shape = (displacements_global.shape[0] - 1, displacements_global.shape[1], 2), dtype = float)
        displacements_projected_samples[:, :, 0] = np.einsum('ctd,cd->ct', displacements_global.samples_time[:-1], path_directions_global)
        displacements_projected_samples[:, :, 1] = np.einsum('ctd,cd->ct', displacements_global.samples_time[1:], path_directions_global)

        displacements_projected = Signal(
            samples = displacements_projected_samples,
            sample_rate = displacements_global.sample_rate
        )

        return_list = results_list[:-1]
        if return_displacements_global: return_list.append(displacements_global)
        return_list.append(displacements_projected)

        logger.debug("Returning projected seismograms")
        return return_list

    @override
    def request_fibre_strains(self,
                path: Path,
                step_length: float = None,
                duration: float = None,
                return_displacements_local: bool = False,
                return_displacements_global: bool = False,
                return_displacements_projected: bool = False,
                batch_size: int = None,
                worker_count: int = 1,
                request_delay: float = 0.1
            ):
        """
        Interpreting longitudes and latitudes as chronological path coordinates, request the longitudinal material strain on each path section by differentiating the projected seismograms in space.

        Inputs:
        - path [Path]: Fibre path with C vertices
        - step_length [float]: if not None, request earthquakes at I points along path spaced step_length apart in km. Then, interpolate the results back to every vertex along path to calculate strains.
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - return_displacements_local [bool]: if True, return displacements in local coordinates in addition to the other return value(s).
        - return_displacements_global [bool]: if True, return displacements in global coordinates in addition to the other return value(s).
        - return_displacements_projected [bool]: if True, return displacements projected onto the fibre in addition to the other return value(s).
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C
        - worker_count [int]: how many Syngine requests to make at most in parallel
        - request_delay [float]: minimum delay in seconds between launching two Syngine requests

        Outputs:
        - If return_displacements_local: [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [I, T, D] with time T, and where D = 3 indexes longitudinal, latitudinal, and normal components in that order.
        - If return_displacements_global: [Signal] signal containing all three displacement components in m, relative to global coordinates, shape [C, T, D] where D = 3 indexes global x, y, and z components in that order.
        - If return_displacements_projected: [Signal] signal containing displacement in m projected onto the path, shape [C-1, T, E] where E = 2 distinguishes between section beginnings and ends.
        - [Signal] signal containing strain projected onto the path, shape [C-1, T, 1].
        """
        results_list = self.request_projected_seismograms(path, step_length, duration, return_displacements_local, return_displacements_global, batch_size, worker_count, request_delay)
        displacements_projected = results_list[-1]

        fibre_strains = Signal(
            samples = np.diff(displacements_projected.samples_time, axis = -1) / path.lengths[:, None, None],
            sample_rate = displacements_projected.sample_rate
        )

        return_list = results_list[:-1]
        if return_displacements_projected: return_list.append(displacements_projected)
        return_list.append(fibre_strains)

        logger.debug("Returning fibre strains")
        return return_list