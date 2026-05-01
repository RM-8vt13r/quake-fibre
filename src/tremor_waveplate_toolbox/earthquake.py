"""
A class that simulates earthquakes along a path on the earth using Syngine.
"""
from configparser import ConfigParser
import logging
from abc import ABC, abstractmethod
from typing import override

import numpy as np
import obspy as op
import obspy.clients.syngine
import obspy.clients.fdsn
from obspy.clients.base import ClientHTTPException

from .signal import Signal
from .path import Path
from .perturbation_event import PerturbationEvent
from .perturbation import Perturbation
from .thread_pool_executor import ThreadPoolExecutor

logger = logging.getLogger()

class Earthquake(PerturbationEvent, ABC):
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

        logger.info("Creating earthquake..")

        super().__init__()

        self._event = parameters.get('EARTHQUAKE', 'event')
        self._model = parameters.get('EARTHQUAKE', 'model')

        self._init_origin()
        self._init_syngine()

    def _init_origin(self):
        """
        Retrieve event information from FDSN as part of the Earthquake construction.
        This function was only tested for the GCMT catalog so far.
        """
        assert ':' in self.event, f"Event id must look like <catalog>:<identifier>, but was {self.event} (did not contain the ':' character)"
        catalog, identifier = self.event.split(':')

        try:
            if catalog == 'GCMT':
                client = op.clients.fdsn.client.Client()
                if not identifier[0].isnumeric(): identifier = identifier[1:]
                event  = client.get_events(
                        starttime = op.UTCDateTime(f'{identifier[:8]}T{identifier[8:12]}00'),
                        endtime   = op.UTCDateTime(f'{identifier[:8]}T{int(identifier[8:12]) + 5:04d}00'),
                        catalog   = catalog
                    )[0]

            else:
                client = op.clients.fdsn.client.Client(catalog)
                event  = client.get_events(eventid = identifier)[0]

        except:
            raise ValueError(f"Could not retrieve event with identifier {self.event} from FDSN server")

        self._origin = event.origins[0]

    def _init_syngine(self):
        """
        Create the Syngine client and do a test run.
        """
        self._syngine_client = op.clients.syngine.Client() # Client to request seismograms from the Syngine webservice
        self._interpolation_order = 1

        try:
            logger.info("Initial Syngine test run:")
            self.request_local_seismograms(Path([0, 1], [0, 1]))
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
        ) -> (Path, int):
        """
        Part of request_local_seismograms() that verifies and prepares the parameters before sending requests to Syngine.
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
            
        return earthquake_path, batch_size

    @abstractmethod
    def _local_seismograms_build_batches(self,
                coordinates: np.ndarray,
                batch_size: int
            ) -> (int, list[np.ndarray], np.ndarray, np.ndarray):
        """
        Part of request_local_seismograms() that builds batches of HTTP requests to send to Syngine

        Inputs:
        - coordinates [np.ndarray]: The coordinates at which to obtain seismograms.
        - batch_size [int]: The number of seismograms to request at once per HTTP request.

        Outputs:
        - [int] The number of batches
        - [list] The coordinates at which to request seismograms, per batch
        - [list] The index of the first coordinate per batch
        - [list] The index of the last coordinate per batch
        """
        # Number of coordinate batches
        batch_count = int(np.ceil(coordinates.shape[0] / batch_size))
        
        # Seismogram coordinates per batch (excluding the last batch, which is added later), with corresponding coordinate indices for logging (including the last batch already)
        batch_coordinates = np.reshape(coordinates[:(batch_count - 1) * batch_size, :], (batch_count - 1, batch_size, 2))
        batch_coordinate_starts = np.arange(batch_count) * batch_size
        batch_coordinate_stops  = np.minimum(batch_coordinate_starts + batch_size, coordinates.shape[0])

        # Convert to list so we can add the last batch, which may contain fewer coordinates than the others
        batch_coordinates = batch_coordinates.tolist()

        # Add the last batch. No need to update the coordinate indices, as these were already generated for this batch.
        batch_coordinates.append(coordinates[(batch_count - 1) * batch_size:])

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

    def _local_seismograms_postprocess(self, seismogram_count: int, syngine_stream: list):
        """
        Part of request_local_seismograms() that verifies and structures the data obtained from Syngine.
        """
        displacements = []
        sample_time = np.mean(np.diff(syngine_stream[0].times()))
        
        times = syngine_stream[0].times()
        assert np.allclose(np.diff(times), sample_time), f"Syngine returned a non-uniform times array"

        displacements = np.zeros(shape = (seismogram_count, len(times), 3), dtype = float)
        for receiver_index in range(seismogram_count):
            assert syngine_stream[3 * receiver_index + 2].id.endswith('MXE'), f"syngine_stream receiver {receiver_index + 1} trace 2 must represent west-east displacement, but doesn't"
            assert syngine_stream[3 * receiver_index + 2].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index + 2].times() == times), f"Receiver {receiver_index + 1} west-east trace times must be the same as receiver 1 normal trace times, but weren't"
            displacements[receiver_index, :, 0] = syngine_stream[3 * receiver_index + 2].data

            assert syngine_stream[3 * receiver_index + 1].id.endswith('MXN'), f"syngine_stream receiver {receiver_index + 1} trace 1 must represent south-north displacement, but doesn't"
            assert syngine_stream[3 * receiver_index + 1].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index + 1].times() == times), f"Receiver {receiver_index + 1} south-north trace times must be the same as receiver 1 normal trace times, but weren't"
            displacements[receiver_index, :, 1] = syngine_stream[3 * receiver_index + 1].data

            assert syngine_stream[3 * receiver_index].id.endswith('MXZ'), f"syngine_stream receiver {receiver_index + 1} trace 0 must represent normal displacement, but doesn't"
            assert syngine_stream[3 * receiver_index].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index].times() == times), f"Receiver {receiver_index + 1} normal trace times must be the same as receiver 1 normal trace times, but weren't"
            displacements[receiver_index, :, 2] = syngine_stream[3 * receiver_index].data

        logger.info("Seismograms postprocessed")

        return displacements, sample_time

    def request_local_seismograms(self,
                path: Path,
                step_length: float = None,
                duration: float = None,
                batch_size: int = None,
                worker_count: int = 1,
                request_delay: float = 0.1
            ) -> (Path, Signal):
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
        earthquake_path, batch_size = self._local_seismograms_preprocess(path, step_length, duration, batch_size, worker_count, request_delay)
        coordinate_batch_count, batch_coordinates, batch_coordinate_starts, batch_coordinate_stops = self._local_seismograms_build_batches(earthquake_path, batch_size)
        syngine_stream = self._local_seismograms_send_requests(duration, coordinate_batch_count, batch_coordinates, batch_coordinate_starts, batch_coordinate_stops, worker_count, request_delay)
        displacements, sample_time = self._local_seismograms_postprocess(earthquake_path, syngine_stream)
        
        displacements_local = Signal(
                samples = displacements,
                sample_rate = 1 / sample_time
            )

        logger.debug("Returning local seismograms")
        return earthquake_path, displacements_local

    # @override
    # def get_perturbations(self,
    #             path: Path,
    #             step_length: float = None,
    #             filter_frequencies: np.ndarray = None,
    #             filter_taps: np.ndarray = None,
    #             duration: float = None,
    #             batch_size: int = None,
    #             worker_count: int = 1,
    #             request_delay: float = 0.1
    #         ) -> tuple[Perturbation]:
    #     """
    #     Create perturbations from this earthquake. See Earthquake.request_projected_strains() and PerturbationEvent.get_perturbation() for documentation details.
    #     """
    #     strains, = self.request_fibre_strains(path, step_length, duration, filter_frequencies, filter_taps, False, False, False, batch_size, worker_count, request_delay)

    #     perturbation = Perturbation(
    #             strains     = strains.samples_time[:, :, 0],
    #             sample_rate = strains.sample_rate,
    #             domain      = strains.domain
    #         )

    #     logger.debug("Returning earthquake perturbation")
    #     return perturbation
    
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
    def origin(self):
        """
        [str] The earthquake origin.
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        raise AttributeError("Cannot change earthquake origin after instantiation; create a new instance instead")

    @property
    def model(self):
        """
        [str] The earth model used by Syngine to synthesise seismograms.
        """
        return self._model

    @model.setter
    def model(self, value):
        raise AttributeError("Cannot change earth model after instantiation; create a new instance instead")