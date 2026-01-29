"""
A class that simulates earthquakes along a path on the earth using Syngine.
"""
from configparser import ConfigParser
import logging
from typing import override

import numpy as np
import scipy as sp
import obspy as op
import obspy.clients.syngine
from obspy.geodetics.base import WGS84_A as earth_radius # meters
from obspy.clients.base import ClientHTTPException

from .signal import Signal
from .path import Path
from .perturbation_event import PerturbationEvent
from .perturbation import Perturbation
from .thread_pool_executor import ThreadPoolExecutor

logger = logging.getLogger()

class Earthquake(PerturbationEvent):
    def __init__(self, parameters: ConfigParser):
        """
        Initialise the earthquake

        Required entries in parameters['EARTHQUAKE']:
        - event [str]: Identifier of a historic earthquake event, e.g. from https://www.globalcmt.org/ or another database. Structure is <catalog>:<identifier>. Example: 'GCMT:C201002270634A' refers to the event at https://ds.iris.edu/spud/momenttensor/987510
        - model [str]: Earth model for Syngine to use from https://ds.iris.edu/ds/products/syngine/#earth. The model dictates at what depth seismograms are synthesised (usually the ocean floor or surface)
        """
        assert 'EARTHQUAKE' in parameters, "Parameters are missing section 'EARTHQUAKE'"
        assert 'event' in parameters['EARTHQUAKE'], "'event' is missing from parameters section 'EARTHQUAKE'."
        assert 'model' in parameters['EARTHQUAKE'], "'model' is missing from parameters section 'EARTHQUAKE'."

        super().__init__()

        self._event = parameters.get('EARTHQUAKE', 'event')
        self._model = parameters.get('EARTHQUAKE', 'model')

        self._syngine_client = op.clients.syngine.Client() # Client to request seismograms from the Syngine webservice

        self._interpolation_order = 1

        try:
            logger.info("Initial Syngine test run:")
            self.request_local_seismograms(Path([0], [0]))
        except ClientHTTPException as e:
            if 'HTTP code 400' in str(e):
                raise ValueError(f"Earth model {self.model} is not known by Syngine; look at https://ds.iris.edu/ds/products/syngine/#earth for available models")
            elif 'HTTP code 204' in str(e):
                raise ValueError(f"Earthquake event {self.event} is not known by Syngine; look at https://ds.iris.edu/spud/momenttensor for available events")
            else:
                raise e

    def _local_seismograms_send_requests_worker(self,
                coordinates: (list, tuple),
                duration: float,
                batch_coordinate_start: int,
                batch_coordinate_stop: int,
                batch_index: int,
                batch_count: int
            ) -> op.Stream:
        """
        Worker function for the parallelisation of _local_seismograms_send_requests(). Requests seismograms for a single batch of coordinates.

        Inputs (arguments):
        - coordinates [list]: list of longitude, latitude coordinates where to request earthquakes, shape [L, 2]
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - batch_coordinate_start [int]: start index of the coordinates in this batch starting at 0
        - batch_coordinate_stop [int]: stop index of the coordinates in this batch starting at 0
        - batch_index [int]: index of this batch starting at 0
        - batch_count [int]: total number of batches that are being requested

        Outputs:
        - [op.Stream] seismograms for the requestse batch
        """
        logger.info(f"Requesting seismograms at coordinates {batch_coordinate_start + 1}-{batch_coordinate_stop} (batch {batch_index + 1} of {batch_count}).")
        kwargs = {
            'model': self.model,
            'bulk': [{
                'longitude': coordinate[0],
                'latitude': coordinate[1]
            } for coordinate in coordinates],
            'eventid': self.event
        }

        if duration is not None:
            kwargs = kwargs | {
                'endtime': duration - 0.0001 # -0.0001 such that the temporal endpoint is not included.
            }

        try:
            return self._syngine_client.get_waveforms_bulk(**kwargs)
        except ClientHTTPException as e:
            if 'HTTP code 204' in str(e):
                error_string = "Sygine could not synthesise "
                if duration is not None:
                    error_string += f"{round(duration, 1)}-second "
                error_string += f"seismograms at coordinates {batch_coordinate_start + 1}-{batch_coordinate_stop} (batch {batch_index + 1} of {batch_count})."
                raise ClientHTTPException(error_string)

    def _local_seismograms_preprocess(self,
        path: Path,
        step_length: float,
        duration: float,
        batch_size: int,
        worker_count: int,
        request_delay: float
    ) -> (Path, np.ndarray, float, int):
        """
        Part of request_local_seismograms() that requests seismograms from Syngine by HTTP request.
        """
        earthquake_path = path if step_length is None else path.interpolated(step_length)

        if batch_size is None:
            batch_size = earthquake_path.vertex_count

        assert isinstance(batch_size, (int, np.integer)), f"batch_size must be an int, but was a {type(batch_size)}"
        assert isinstance(worker_count, (int, np.integer)), f"worker_count must be an int, but was a {type(worker_count)}"
        assert worker_count >= 1, f"worker_count must be >= 1, but was {worker_count}"
        assert request_delay >= 0, f"request_delay must be >= 0, but was {request_delay}"
        if duration is not None:
            assert isinstance(duration, (int, np.integer, float, np.floating)), f"duration should have type float, but was a {type(duration)}"
            assert duration > 0, f"duration must be positive, but wasn't"
            
        return earthquake_path, duration, batch_size

    def _local_seismograms_build_batches(self,
                earthquake_path: Path,
                batch_size: int
            ) -> (int, list[np.ndarray], np.ndarray, np.ndarray):
        """
        Part of request_local_seismograms() that builds batches of HTTP requests to send to Syngine
        """
        # Number of coordinate batches
        batch_count = int(np.ceil(earthquake_path.vertex_count / batch_size))
        
        # Seismogram coordinates per batch (excluding the last batch, which is added later), with corresponding coordinate indices for logging (including the last batch already)
        batch_coordinates = np.reshape(earthquake_path.coordinates[:(batch_count - 1) * batch_size], (batch_count - 1, batch_size, 2))
        batch_coordinate_starts = np.arange(batch_count) * batch_size
        batch_coordinate_stops  = np.minimum(batch_coordinate_starts + batch_size, earthquake_path.vertex_count)

        # Convert to list so we can add the last batch, which may contain fewer coordinates than the others
        batch_coordinates = batch_coordinates.tolist()

        # Add the last batch. No need to update the coordinate indices, as these were already generated for this batch.
        batch_coordinates.append(earthquake_path.coordinates[(batch_count - 1) * batch_size:])

        return batch_count, batch_coordinates, batch_coordinate_starts, batch_coordinate_stops

    def _local_seismograms_send_requests(self,
                duration: float,
                batch_count: int,
                batch_coordinates: np.ndarray,
                batch_coordinate_starts: np.ndarray,
                batch_coordinate_stops: np.ndarray,
                worker_count: int,
                request_delay: float
            ) -> list[op.Stream]:
        """
        Part of request_local_seismograms() that requests seismograms from Syngine by HTTP request.
        """
        log_string = "Requesting "
        if duration is not None:
            log_string += f"{round(duration, 1)}-second "
        log_string += f"seismograms from Syngine at {batch_coordinate_stops[-1]} coordinate(s) using {worker_count} worker(s)"
        logger.info(log_string)

        try:
            # Create a new Syngine stream
            syngine_stream = op.core.stream.Stream()
            
            # Start threads to request seismograms concurrently
            with ThreadPoolExecutor(max_workers = worker_count) as pool:
                batch_streams = pool.map(
                        self._local_seismograms_send_requests_worker,
                        batch_coordinates,
                        [duration] * batch_count,
                        batch_coordinate_starts,
                        batch_coordinate_stops,
                        np.arange(batch_count),
                        [batch_count] * batch_count,
                        timeout = None,
                        buffersize = 3 * worker_count,
                        jobdelay = request_delay
                    )

                for batch_stream in batch_streams:
                    syngine_stream += batch_stream

        except ClientHTTPException as e:
            if 'HTTP code 404' in str(e):
                raise ValueError(f"Syngine cannot be reached, try again later. You were possibly blocked temporarily, if your worker_count or request_delay violate the guidelines at https://ds.iris.edu/ds/nodes/dmc/services/usage/.")
            else:
                raise e
        except ConnectionError as e:
            raise ConnectionError("Syngine cannot be be reached, check your internet connection")

        log_string = ""
        if duration is not None:
            log_string += f"{round(duration, 1)}-second "
        log_string = "seismograms obtained"
        logger.info(log_string)

        return syngine_stream

    def _local_seismograms_postprocess(self, earthquake_path: Path, syngine_stream: list):
        """
        Part of request_local_seismograms() that verifies and structures the data obtained from Syngine.
        """
        # times = np.zeros(shape = (0,), dtype = float)
        # displacements = np.zeros(shape = (earthquake_path.vertex_count, 0), dtype = float)
        displacements = []
        sample_time = np.mean(np.diff(syngine_stream[0].times()))
        
        times = syngine_stream[0].times()
        assert np.allclose(np.diff(times), sample_time), f"Syngine returned a non-uniform times array"

        displacements = np.zeros(shape = (earthquake_path.vertex_count, len(times), 3), dtype = float)
        # displacements_longitude = np.zeros(shape = (earthquake_path.vertex_count, len(times)), dtype = float)
        # displacements_latitude  = np.zeros_like(displacements_longitude)
        # displacements_normal    = np.zeros_like(displacements_longitude)
        for receiver_index in range(earthquake_path.vertex_count):
            assert syngine_stream[3 * receiver_index + 2].id.endswith('MXE'), f"syngine_stream receiver {receiver_index + 1} trace 2 must represent west-east displacement, but doesn't"
            assert syngine_stream[3 * receiver_index + 2].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index + 2].times() == times), f"Receiver {receiver_index + 1} west-east trace times must be the same as receiver 1 normal trace times, but weren't"
            # displacements_longitude[receiver_index, :] = syngine_stream[3 * receiver_index + 2].data
            displacements[receiver_index, :, 0] = syngine_stream[3 * receiver_index + 2].data

            assert syngine_stream[3 * receiver_index + 1].id.endswith('MXN'), f"syngine_stream receiver {receiver_index + 1} trace 1 must represent south-north displacement, but doesn't"
            assert syngine_stream[3 * receiver_index + 1].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index + 1].times() == times), f"Receiver {receiver_index + 1} south-north trace times must be the same as receiver 1 normal trace times, but weren't"
            # displacements_latitude[receiver_index, :] = syngine_stream[3 * receiver_index + 1].data
            displacements[receiver_index, :, 1] = syngine_stream[3 * receiver_index + 1].data

            assert syngine_stream[3 * receiver_index].id.endswith('MXZ'), f"syngine_stream receiver {receiver_index + 1} trace 0 must represent normal displacement, but doesn't"
            assert syngine_stream[3 * receiver_index].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index].times() == times), f"Receiver {receiver_index + 1} normal trace times must be the same as receiver 1 normal trace times, but weren't"
            # displacements_normal[receiver_index, :] = syngine_stream[3 * receiver_index].data
            displacements[receiver_index, :, 2] = syngine_stream[3 * receiver_index].data

        # displacements = np.stack([displacements_longitude, displacements_latitude, displacements_normal], axis = -1)
        
        logger.info("Seismograms postprocessed")

        return displacements, sample_time

    def request_local_seismograms(self,
                path: Path,
                step_length: float = None,
                duration: float = None,
                batch_size: int = None,
                worker_count: int = 1,
                request_delay: float = 0.1
            ):
        """
        Request seismograms from Syngine at specified coordinates, on the local (longitude, latitude, normal) axes at each coordinate.

        Inputs:
        - path [Path]: coordinates, length C
        - step_length [float]: if not None, request earthquakes at I points along path spaced step_length apart in km.
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C
        - worker_count [int]: how many Syngine requests to make at most in parallel
        - request_delay [float]: minimum delay in seconds between launching two Syngine requests
        
        Outputs:
        - [Path] interpolated version of path with vertices spaced step_length km apart
        - [Signal] signal containing all three displacement components in m, shape [I, T, D] where D indexes longitudinal, latitudinal, and normal components in that order.
        """
        earthquake_path, duration, batch_size = self._local_seismograms_preprocess(path, step_length, duration, batch_size, worker_count, request_delay)
        coordinate_batch_count, batch_coordinates, batch_coordinate_starts, batch_coordinate_stops = self._local_seismograms_build_batches(earthquake_path, batch_size)
        syngine_stream = self._local_seismograms_send_requests(duration, coordinate_batch_count, batch_coordinates, batch_coordinate_starts, batch_coordinate_stops, worker_count, request_delay)
        displacements, sample_time = self._local_seismograms_postprocess(earthquake_path, syngine_stream)
        
        displacements_local = Signal(
                samples = displacements,
                sample_rate = 1 / sample_time
            )

        logger.debug("Returning local seismograms")
        return earthquake_path, displacements_local
    
    def _global_seismograms_interpolate(self, earthquake_path: Path, path: Path, displacements: np.ndarray):
        """
        Part of request_global_seismograms() that interpolates sparsely obtained seismograms to a denser path.
        """
        # interpolator  = sp.interpolate.FloaterHormannInterpolator(earthquake_path.positions, displacements)
        
        # interpolator = sp.interpolate.make_interp_spline(earthquake_path.positions, displacements, k = self._interpolation_order)
        # displacements_interpolated = interpolator(path.positions)
        displacements_interpolated_flattened = np.zeros(shape = (np.prod(displacements.shape[-2:]), path.vertex_count)) # [T * D, C]
        displacements_flattened = displacements.reshape((earthquake_path.vertex_count, np.prod(displacements.shape[-2:]))).transpose() # [I, T, D] -> [I, T * D] -> [T * D, I]

        for channel_index, displacement_flattened in enumerate(displacements_flattened):
            displacements_interpolated_flattened[channel_index] = np.interp(path.positions, earthquake_path.positions, displacement_flattened)

        displacements_interpolated = displacements_interpolated_flattened.transpose().reshape((path.vertex_count, *displacements.shape[-2:])) # [T * D, C] -> [C, T * D] -> [C, T, D]

        # except ValueError as e:
        #     raise ValueError()

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
        Request seismograms from Syngine at specified coordinates, and transform them from the local axes at each coordinate to a shared global axes system.

        Inputs:
        - path [Path]: coordinates, length C
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

        # if step_length is not None:
        #     assert len(earthquake_path) > self._interpolation_order, f"Ground displacements at at least {self._interpolation_order} positions are necessary for interpolation, but step_length {step_length} yielded displacements at only {len(earthquake_path)} positions"

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
        - path [Path]: coordinates, length C
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

        # displacements_vertices = np.stack([
        #     displacements_global.samples_time[:-1],
        #     displacements_global.samples_time[1:]
        # ], axis = -1) # [C - 1, T, D, E = 2]

        displacements_projected_samples = np.zeros(shape = (displacements_global.shape[0] - 1, displacements_global.shape[1], 2), dtype = float)
        displacements_projected_samples[:, :, 0] = np.einsum('ctd,cd->ct', displacements_global.samples_time[:-1], path_directions_global)
        displacements_projected_samples[:, :, 1] = np.einsum('ctd,cd->ct', displacements_global.samples_time[1:], path_directions_global)

        # displacements_projected = Signal(
        #     samples = np.einsum('ctde,cd->cte', displacements_vertices, path_directions_global),
        #     sample_rate = displacements_global.sample_rate
        # )
        displacements_projected = Signal(
            samples = displacements_projected_samples,
            sample_rate = displacements_global.sample_rate
        )

        return_list = results_list[:-1]
        if return_displacements_global: return_list.append(displacements_global)
        return_list.append(displacements_projected)

        logger.debug("Returning projected seismograms")
        return return_list

    def request_projected_strains(self,
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
        - path [Path]: coordinates, length C
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

        strains_projected = Signal(
            samples = np.diff(displacements_projected.samples_time, axis = -1) / path.lengths[:, None, None],
            sample_rate = displacements_projected.sample_rate
        )

        return_list = results_list[:-1]
        if return_displacements_projected: return_list.append(displacements_projected)
        return_list.append(strains_projected)

        logger.debug("Returning projected strains")
        return return_list

    @override
    def get_perturbations(self,
                path: Path,
                step_length: float = None,
                duration: float = None,
                batch_size: int = None,
                worker_count: int = 1,
                request_delay: float = 0.1
            ) -> tuple[Perturbation]:
        """
        Create perturbations from this earthquake. See Earthquake.request_projected_strains() and PerturbationEvent.get_perturbation() for documentation details.
        """
        strains_projected, = self.request_projected_strains(path, step_length, duration, False, False, False, batch_size, worker_count, request_delay)

        perturbation = Perturbation(
                strains     = strains_projected.samples_time[:, :, 0],
                sample_rate = strains_projected.sample_rate,
                domain      = strains_projected.domain
            )

        logger.debug("Returning earthquake perturbation")
        return perturbation

    @property
    def event(self):
        """
        [str] The event identifier of this earthquake.
        """
        return self._event

    @event.setter
    def event(self, value):
        raise AttributeError("Cannot change earthquake event ID after instantiation; create a new instance instead")

    @property
    def model(self):
        """
        [str] The earth model used by Syngine to synthesise seismograms.
        """
        return self._model

    @model.setter
    def model(self, value):
        raise AttributeError("Cannot change earth model after instantiation; create a new instance instead")