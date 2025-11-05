"""
A class that simulates earthquakes along an optical fibre using Syngine
"""
from configparser import ConfigParser

import numpy as np
import obspy as op
import obspy.clients.syngine
from obspy.geodetics.base import WGS84_A as earth_radius # m
from obspy.clients.base import ClientHTTPException

from .fibre import Fibre
from .signal import Signal

class Earthquake:
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

        self._event = parameters.get('EARTHQUAKE', 'event')
        self._model = parameters.get('EARTHQUAKE', 'model')

        self._syngine_client = op.clients.syngine.Client() # Client to request seismograms from the Syngine webservice

        try:
            self.request_seismograms([0], [0])
        except ClientHTTPException as e:
            if 'HTTP code 400' in str(e):
                raise ValueError(f"Earth model {self.model} is not known by Syngine; look at https://ds.iris.edu/ds/products/syngine/#earth for available models")
            elif 'HTTP code 204' in str(e):
                raise ValueError(f"Earthquake event {self.event} is not known by Syngine; look at https://ds.iris.edu/spud/momenttensor for available events")
            else:
                raise e

    def __call__(self, fibre: Fibre, batch_size: int = 100, verbose: bool = False):
        """
        See request_fibre_section_projected_strain()
        """
        return self.request_fibre_section_projected_strain(fibre, batch_size, verbose)

    def request_seismograms(self, longitudes: np.ndarray, latitudes: np.ndarray, batch_size: int = 100, verbose: bool = False):
        """
        Request seismograms from Syngine at specified coordinates.

        Inputs:
        - longitudes [np.ndarray]: coordinate longitudes, length C
        - latitude [np.ndarray]: coordinate latitudes, length C
        - batch_size [int]: how many seismograms to request simultaneously
        - verbose: whether to print requesting progress

        Outputs:
        - [Signal] signal containing all three displacement components in m, shape [C, T, D] where D indexes longitudinal, latitudinal, and normal components in that order
        """
        longitudes, latitudes = np.array(longitudes), np.array(latitudes)
        assert len(longitudes.shape) == 1, f"longitudes must have one dimension, but had {len(longitudes.shape)} ({longitudes.shape})"
        assert latitudes.shape == longitudes.shape, f"longitudes and latitudes must have the same shape, but had shapes {longitudes.shape} and {latitudes.shape}"

        if verbose: print(f"Requesting seismograms from Syngine at {len(longitudes)} coordinates along the fibre. This may take a while..")
        try:
            batch_count = int(np.ceil(longitudes.shape[0] / batch_size))

            syngine_stream = op.core.stream.Stream()
            for batch_index in range(batch_count):
                batch_start = batch_index * batch_size
                batch_stop  = min(batch_start + batch_size, longitudes.shape[0])
                if verbose: print(f"Requesting seismograms at coordinates {batch_start + 1}-{batch_stop} (batch {batch_index + 1} of {batch_count}).")
                syngine_stream += self._syngine_client.get_waveforms_bulk(
                    model = self.model,
                    bulk = [{
                        'longitude': longitude,
                        'latitude': latitude
                    } for longitude, latitude in zip(longitudes[batch_start:batch_stop], latitudes[batch_start:batch_stop])],
                    eventid = self.event
                )
                # import pdb
                # pdb.set_trace()

        except ClientHTTPException as e:
            if 'HTTP code 404' in str(e):
                raise ValueError(f"Syngine can currently not be reached; try again later")
            else:
                raise e
        except ConnectionError as e:
            raise ConnectionError("Syngine could not be reached; please check your internet connection")
        
        if verbose: print("Syngine request complete. Collecting and validating results..")

        times = syngine_stream[0].times()
        sample_time = times[1] - times[0]
        assert np.allclose(np.diff(times), sample_time), f"Syngine returned a non-uniform times array"

        displacements_longitude = np.zeros(shape = (len(longitudes), len(times)), dtype = float)
        displacements_latitude  = np.zeros_like(displacements_longitude)
        displacements_normal    = np.zeros_like(displacements_longitude)
        for receiver_index in range(len(longitudes)):
            assert syngine_stream[3 * receiver_index + 2].id.endswith('MXE'), f"syngine_stream receiver {receiver_index + 1} trace 2 must represent west-east displacement, but doesn't"
            assert syngine_stream[3 * receiver_index + 2].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index + 2].times() == times), f"Receiver {receiver_index + 1} west-east trace times must be the same as receiver 1 normal trace times, but weren't"
            displacements_longitude[receiver_index, :] = syngine_stream[3 * receiver_index + 2].data

            assert syngine_stream[3 * receiver_index + 1].id.endswith('MXN'), f"syngine_stream receiver {receiver_index + 1} trace 1 must represent south-north displacement, but doesn't"
            assert syngine_stream[3 * receiver_index + 1].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index + 1].times() == times), f"Receiver {receiver_index + 1} south-north trace times must be the same as receiver 1 normal trace times, but weren't"
            displacements_latitude[receiver_index, :] = syngine_stream[3 * receiver_index + 1].data

            assert syngine_stream[3 * receiver_index].id.endswith('MXZ'), f"syngine_stream receiver {receiver_index + 1} trace 0 must represent normal displacement, but doesn't"
            assert syngine_stream[3 * receiver_index].times().shape == times.shape and np.all(syngine_stream[3 * receiver_index].times() == times), f"Receiver {receiver_index + 1} normal trace times must be the same as receiver 1 normal trace times, but weren't"
            displacements_normal[receiver_index, :] = syngine_stream[3 * receiver_index].data

        if verbose: print("Results validated and returned!")

        return Signal(
            samples = np.stack([
                displacements_longitude,
                displacements_latitude,
                displacements_normal
            ], axis = -1),
            sample_rate = 1 / sample_time
        )

    def request_fibre_path_seismograms(self, fibre: Fibre, batch_size: int = 100, verbose: bool = False):
        """
        Request seismograms from Syngine at fibre path segment coordinates.

        Inputs:
        - fibre: the fibre at which sections to request earthquakes. The fibre must be initialised with path coordinates.
        - batch_size [int]: how many seismograms to request simultaneously
        - verbose: whether to print requesting progress

        Outputs:
        - [Signal] signal containing all three displacement components in m, shape [P+1, T, D] with number of path segments P and where D = 3 indexes longitudinal, latitudinal, and normal components in that order
        """
        return self.request_seismograms(*fibre.path_coordinates.T, batch_size, verbose)

    def request_fibre_section_seismograms(self, fibre: Fibre, batch_size: int = 100, verbose: bool = False):
        """
        Request seismograms from Syngine at fibre section coordinates.

        Inputs:
        - fibre: the fibre at which sections to request earthquakes. The fibre must be initialised with path coordinates.
        - batch_size [int]: how many seismograms to request simultaneously
        - verbose: whether to print requesting progress

        Outputs:
        - [Signal] signal containing all three displacement components in m, shape [S+1, T, D] with number of fibre sections S and where D = 3 indexes longitudinal, latitudinal, and normal components in that order
        """
        return self.request_seismograms(*fibre.section_coordinates.T, batch_size, verbose)

    def request_fibre_section_projected_seismogram(self, fibre: Fibre, batch_size: int = 100, verbose: bool = False):
        """
        Request seismograms from Syngine at fibre section coordinates, projected onto the direction of each fibre section.

        Inputs:
        - fibre: the fibre at which sections to request earthquakes. The fibre must be initialised with path coordinates.
        - batch_size [int]: how many seismograms to request simultaneously
        - verbose: whether to print requesting progress

        Outputs:
        - [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [S+1, T, D] with number of fibre sections S, time T, and where D = 3 indexes longitudinal, latitudinal, and normal components in that order
        - [Signal] signal containing all three displacement components in m, relative to global coordinates, shape [S+1, T, D] where D = 3 indexes x, y, and z components in that order
        - [Signal] signal containing displacement in m projected onto the fibre, shape [S, T, E] where E = 2 distinguighes between section beginnings and ends
        """
        displacements_local = self.request_fibre_section_seismograms(fibre, batch_size, verbose)

        # Calculate Cartesian global fibre section direction vectors
        section_endpoints_longitudes, section_endpoints_latitudes = fibre.section_coordinates.T
        section_endpoints_global = earth_radius * np.array([
            np.cos(section_endpoints_latitudes) * np.cos(section_endpoints_longitudes),
            np.cos(section_endpoints_latitudes) * np.sin(section_endpoints_longitudes),
            np.sin(section_endpoints_latitudes)
        ]).T # [S+1, C = 3]

        section_directions_global = section_endpoints_global[1:] - section_endpoints_global[:-1] # [S, G = 3]
        section_directions_global /= np.linalg.norm(section_directions_global, axis = -1)[:, None]

        # Calculate Cartesian global earthquake components
        sin_long = np.sin(section_endpoints_longitudes)
        sin_lat  = np.sin(section_endpoints_latitudes)
        cos_long = np.cos(section_endpoints_longitudes)
        cos_lat  = np.cos(section_endpoints_latitudes)
        zeros    = np.zeros_like(sin_long)

        axes_local = np.array([
            [-sin_long, -sin_lat * cos_long, cos_lat * cos_long],
            [ cos_long, -sin_lat * sin_long, cos_lat * sin_long],
            [ zeros   ,  cos_lat           , sin_lat           ]
        ]).transpose((2, 0, 1)) # [S+1, G, L = 3]

        displacements_global = Signal(
            samples = np.einsum('sgl,stl->stg', axes_local, displacements_local.samples_time),
            sample_rate = displacements_local.sample_rate
        )
        
        # Project the earthquakes onto the fibre directions
        displacements_ends = np.stack([
            displacements_global.samples_time[:-1],
            displacements_global.samples_time[1:]
        ], axis = 3) # [S, T, G, E = 2]

        displacements_projected = Signal(
            samples = np.einsum('stge,sg->ste', displacements_ends, section_directions_global),
            sample_rate = displacements_local.sample_rate
        )

        return displacements_local, displacements_global, displacements_projected
        
    def request_fibre_section_projected_strain(self, fibre: Fibre, batch_size: int = 100, verbose: bool = False):
        """
        Request seismograms from Syngine at fibre section coordinates, projected onto the direction of each fibre section.
        Differentiate the projected seismogram to obtain material strain.

        Inputs:
        - fibre: the fibre at which sections to request earthquakes. The fibre must be initialised with path coordinates.
        - batch_size [int]: how many seismograms to request simultaneously
        - verbose: whether to print requesting progress

        Outputs:
        - [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [S+1, T, D] with number of fibre sections S, time T, and where D = 3 indexes longitudinal, latitudinal, and normal components in that order
        - [Signal] signal containing all three displacement components in m, relative to global coordinates, shape [S+1, T, D] where D = 3 indexes x, y, and z components in that order
        - [Signal] signal containing displacement in m projected onto the fibre, shape [S, T, E] where E = 2 distinguighes between section beginnings and ends
        - [Signal] signal containing strain projected onto the fibre, shape [S, T, 1]
        """
        displacements_local, displacements_global, displacements_projected = self.request_fibre_section_projected_seismogram(fibre, batch_size, verbose)

        strains_projected = Signal(
            samples = np.diff(displacements_projected.samples_time, axis = 2) / fibre.section_lengths[:, None, None],
            sample_rate = displacements_projected.sample_rate
        )

        return displacements_local, displacements_global, displacements_projected, strains_projected

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