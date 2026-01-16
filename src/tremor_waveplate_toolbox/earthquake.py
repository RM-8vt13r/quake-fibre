"""
A class that simulates earthquakes along a path on the earth using Syngine.
"""
from configparser import ConfigParser
import logging
from typing import override

import numpy as np
import obspy as op
import obspy.clients.syngine
from obspy.geodetics.base import WGS84_A as earth_radius # m
from obspy.clients.base import ClientHTTPException

from .signal import Signal
from .path import Path
from .perturbation_event import PerturbationEvent
from .perturbation import Perturbation

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

        try:
            self.request_local_seismograms(Path([0], [0]))
        except ClientHTTPException as e:
            if 'HTTP code 400' in str(e):
                raise ValueError(f"Earth model {self.model} is not known by Syngine; look at https://ds.iris.edu/ds/products/syngine/#earth for available models")
            elif 'HTTP code 204' in str(e):
                raise ValueError(f"Earthquake event {self.event} is not known by Syngine; look at https://ds.iris.edu/spud/momenttensor for available events")
            else:
                raise e

    def request_local_seismograms(self, path: Path, batch_size: int = None):
        """
        Request seismograms from Syngine at specified coordinates, on the local (longitude, latitude, normal) axes at each coordinate.

        Inputs:
        - path [Path]: coordinates, length C
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C
        
        Outputs:
        - [Signal] signal containing all three displacement components in m, shape [C, T, D] where D indexes longitudinal, latitudinal, and normal components in that order
        """
        if batch_size is None: batch_size = path.vertex_count
        assert isinstance(batch_size, int), f"batch_size must be an int, but was a {type(batch_size)}"

        logger.info(f"Requesting seismograms from Syngine at {path.vertex_count} coordinates along the path. This may take a while..")
        try:
            batch_count = int(np.ceil(path.vertex_count / batch_size))

            syngine_stream = op.core.stream.Stream()
            for batch_index in range(batch_count):
                batch_start = batch_index * batch_size
                batch_stop  = min(batch_start + batch_size, path.vertex_count)
                if batch_size != path.vertex_count: logger.info(f"Requesting seismograms at coordinates {batch_start + 1}-{batch_stop} (batch {batch_index + 1} of {batch_count}).")
                syngine_stream += self._syngine_client.get_waveforms_bulk(
                    model = self.model,
                    bulk = [{
                        'longitude': longitude,
                        'latitude': latitude
                    } for (longitude, latitude) in path.coordinates[batch_start:batch_stop]],
                    eventid = self.event
                )

        except ClientHTTPException as e:
            if 'HTTP code 404' in str(e):
                raise ValueError(f"Syngine can currently not be reached; try again later")
            else:
                raise e
        except ConnectionError as e:
            raise ConnectionError("Syngine could not be reached; please check your internet connection")
        
        logger.info("Syngine request complete. Collecting and validating results..")

        times = syngine_stream[0].times()
        sample_time = times[1] - times[0]
        assert np.allclose(np.diff(times), sample_time), f"Syngine returned a non-uniform times array"

        displacements_longitude = np.zeros(shape = (path.vertex_count, len(times)), dtype = float)
        displacements_latitude  = np.zeros_like(displacements_longitude)
        displacements_normal    = np.zeros_like(displacements_longitude)
        for receiver_index in range(path.vertex_count):
            assert syngine_stream[3 * receiver_index + 2].id.endswith('MXE'), f"syngine_stream receiver {receiver_index + 1} trace 2 must represent west-east displacement, but doesn't"
            assert syngine_stream[3 * receiver_index + 2].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index + 2].times() == times), f"Receiver {receiver_index + 1} west-east trace times must be the same as receiver 1 normal trace times, but weren't"
            displacements_longitude[receiver_index, :] = syngine_stream[3 * receiver_index + 2].data

            assert syngine_stream[3 * receiver_index + 1].id.endswith('MXN'), f"syngine_stream receiver {receiver_index + 1} trace 1 must represent south-north displacement, but doesn't"
            assert syngine_stream[3 * receiver_index + 1].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index + 1].times() == times), f"Receiver {receiver_index + 1} south-north trace times must be the same as receiver 1 normal trace times, but weren't"
            displacements_latitude[receiver_index, :] = syngine_stream[3 * receiver_index + 1].data

            assert syngine_stream[3 * receiver_index].id.endswith('MXZ'), f"syngine_stream receiver {receiver_index + 1} trace 0 must represent normal displacement, but doesn't"
            assert syngine_stream[3 * receiver_index].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index].times() == times), f"Receiver {receiver_index + 1} normal trace times must be the same as receiver 1 normal trace times, but weren't"
            displacements_normal[receiver_index, :] = syngine_stream[3 * receiver_index].data

        logger.info("Results validated and returned!")

        return Signal(
            samples = np.stack([
                displacements_longitude,
                displacements_latitude,
                displacements_normal
            ], axis = -1),
            sample_rate = 1 / sample_time
        )

    def request_global_seismograms(self, path: Path, batch_size: int = None):
        """
        Request seismograms from Syngine at specified coordinates, and transform them from the local axes at each coordinate to a shared global axes system.

        Inputs:
        - path [Path]: coordinates, length C
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C

        Outputs:
        - [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [C, T, D] with time T, and where D = 3 indexes longitudinal, latitudinal, and normal components in that order
        - [Signal] signal containing all three displacement components in m, relative to global coordinates, shape [C, T, D] where D = 3 indexes global x, y, and z components in that order
        """
        displacements_local = self.request_local_seismograms(path, batch_size)

        sin_long = np.sin(path.longitudes)
        sin_lat  = np.sin(path.latitudes)
        cos_long = np.cos(path.longitudes)
        cos_lat  = np.cos(path.latitudes)
        zeros    = np.zeros_like(sin_long)

        transformation_global_to_local = np.array([
            [-sin_long, -sin_lat * cos_long, cos_lat * cos_long],
            [ cos_long, -sin_lat * sin_long, cos_lat * sin_long],
            [ zeros   ,  cos_lat           , sin_lat           ]
        ]).transpose((2, 0, 1)) # [S+1, G, L = 3]

        displacements_global = Signal(
            samples = np.einsum('cgl,ctl->ctg', transformation_global_to_local, displacements_local.samples_time),
            sample_rate = displacements_local.sample_rate
        )

        return displacements_local, displacements_global

    def request_projected_seismograms(self, path: Path, batch_size: int = None):
        """
        Interpreting longitudes and latitudes as chronological path coordinates, project seismograms at each coordinate onto the neighbouring straight segments of this path.
 
        Inputs:
        - path [Path]: coordinates, length C
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C

        Outputs:
        - [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [C, T, D] with time T, and where D = 3 indexes longitudinal, latitudinal, and normal components in that order
        - [Signal] signal containing all three displacement components in m, relative to global coordinates, shape [C, T, D] where D = 3 indexes global x, y, and z components in that order
        - [Signal] signal containing displacement in m projected onto the path, shape [C-1, T, E] where E = 2 distinguishes between section beginnings and ends
        """
        displacements_local, displacements_global = self.request_global_seismograms(path, batch_size)

        path_coordinates_global = earth_radius * np.array([
            np.cos(path.latitudes) * np.cos(path.longitudes),
            np.cos(path.latitudes) * np.sin(path.longitudes),
            np.sin(path.latitudes)
        ]).T

        path_directions_global = path_coordinates_global[1:] - path_coordinates_global[:-1]
        path_directions_global /= np.linalg.norm(path_directions_global, axis = -1)[:, None]

        displacements_vertices = np.stack([
            displacements_global.samples_time[:-1],
            displacements_global.samples_time[1:]
        ], axis = 3) # [S, T, G, E = 2]

        displacements_projected = Signal(
            samples = np.einsum('stge,sg->ste', displacements_vertices, path_directions_global),
            sample_rate = displacements_local.sample_rate
        )

        return displacements_local, displacements_global, displacements_projected

    def request_projected_strains(self, path: Path, batch_size: int = None):
        """
        Interpreting longitudes and latitudes as chronological path coordinates, request the longitudinal material strain on each path section by differentiating the projected seismograms in space.

        Inputs:
        - path [Path]: coordinates, length C
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C

        Outputs:
        - [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [C, T, D] with time T, and where D = 3 indexes longitudinal, latitudinal, and normal components in that order
        - [Signal] signal containing all three displacement components in m, relative to global coordinates, shape [C, T, D] where D = 3 indexes global x, y, and z components in that order
        - [Signal] signal containing displacement in m projected onto the path, shape [C-1, T, E] where E = 2 distinguighes between section beginnings and ends
        - [Signal] signal containing strain projected onto the path, shape [C-1, T, 1]
        """
        displacements_local, displacements_global, displacements_projected = self.request_projected_seismograms(path, batch_size)

        strains_projected = Signal(
            samples = np.diff(displacements_projected.samples_time, axis = 2) / path.lengths[:, None, None],
            sample_rate = displacements_projected.sample_rate
        )

        return displacements_local, displacements_global, displacements_projected, strains_projected

    @override
    def get_perturbation(self, path: Path, batch_size: int = None) -> Perturbation:
        """
        Create a perturbation from this earthquake. See Earthquake.request_projected_strains() and PerturbationEvent.get_perturbation() for documentation details.
        """
        _, _, _, strains_projected = self.request_projected_strains(path, batch_size)
        perturbation = Perturbation(
                material_strains = strains_projected.samples_time[:, :, 0],
                sample_rate = strains_projected.sample_rate,
                domain = strains_projected.domain
            )
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