"""
A class that simulates earthquakes along an optical fibre using Syngine
"""
from configparser import ConfigParser

import numpy as np
import obspy as op
import obspy.clients.syngine
from obspy.clients.base import ClientHTTPException

from tremor_waveplate_toolbox import Fibre

class Earthquake:
    def __init__(self, parameters: ConfigParser): # event: str, model: str = 'ak135f_5s'):
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
            self.request_seismogram(0, 0)
        except ClientHTTPException as e:
            if 'HTTP code 400' in str(e):
                raise ValueError(f"Earth model {self.model} is not known by Syngine; look at https://ds.iris.edu/ds/products/syngine/#earth for available models")
            elif 'HTTP code 204' in str(e):
                raise ValueError(f"Earthquake event {self.event} is not known by Syngine; look at https://ds.iris.edu/spud/momenttensor for available events")
            else:
                raise e

    def __call__(self, fibre: Fibre, verbose: bool = False):
        """
        See request_fibre_section_projected_strain()
        """
        return self.request_fibre_section_projected_strain(fibre, verbose)

    def request_seismogram(self, longitude: float, latitude: float):
        """
        Request seismograms from Syngine at a single specified coordinate.

        Inputs:
        - longitude [float]: coordinate longitude
        - latitude [float]: coordinate latitude
        
        Outputs:
        - [np.ndarray] sample timestamps in s
        - [np.ndarray] normal ground displacement over time in m
        - [np.ndarray] longitude (east-west) ground displacement over time in m
        - [np.ndarray] latitude (north-south) ground displacement over time in m
        """
        try:
            syngine_stream = self._syngine_client.get_waveforms(
                model = self.model,
                receiverlongitude = longitude,
                receiverlatitude = latitude,
                eventid = self.event
            )
        except ClientHTTPException as e:
            if 'HTTP code 404' in str(e):
                raise ValueError(f"Syngine can currently not be reached; try again later")
            else:
                raise e
        except ConnectionError as e:
            raise ConnectionError("Syngine could not be reached; please check your internet connection")

        assert syngine_stream[0].id.endswith('MXZ'), "syngine_stream trace 0 must represent normal displacement, but doesn't"
        times = syngine_stream[0].times()
        displacement_normal = syngine_stream[0].data
        assert syngine_stream[1].id.endswith('MXN'), "syngine_stream trace 1 must represent north-south displacement, but doesn't"
        assert syngine_stream[1].times().shape == times.shape and np.all(syngine_stream[1].times() == times), f"North-South trace times must be the same as Normal trace times, but weren't"
        displacement_latitude = syngine_stream[1].data
        assert syngine_stream[2].id.endswith('MXE'), "syngine_stream trace 1 must represent north-south displacement, but doesn't"
        assert syngine_stream[2].times().shape == times.shape and np.all(syngine_stream[2].times() == times), f"East-West trace times must be the same as Normal and North-South trace times, but weren't"
        displacement_longitude = syngine_stream[2].data

        return times, displacement_normal, displacement_longitude, displacement_latitude

    def request_seismograms(self, longitudes: np.ndarray, latitudes: np.ndarray, verbose: bool = False):
        """
        Request seismograms from Syngine at specified coordinates.

        Inputs:
        - longitudes [np.ndarray]: coordinate longitudes, length C
        - latitude [np.ndarray]: coordinate latitudes, length C
        - verbose: whether to print requesting progress

        Outputs:
        - [np.ndarray] sample timestamps in s, length T
        - [np.ndarray] normal ground displacement over time in m, shape [C, T]
        - [np.ndarray] longitude (east-west) ground displacement over time in m, shape [C, T]
        - [np.ndarray] latitude (north-south) ground displacement over time in m, shape [C, T]
        """
        longitudes, latitudes = np.array(longitudes), np.array(latitudes)
        assert len(longitudes.shape) == 1, f"longitudes must have one dimension, but had {len(longitudes.shape)} ({longitudes.shape})"
        assert latitudes.shape == longitudes.shape, f"longitudes and latitudes must have the same shape, but had shapes {longitudes.shape} and {latitudes.shape}"

        coordinates_iterable = zip(longitudes, latitudes)
        if verbose:
            from tqdm import tqdm
            coordinates_iterable = tqdm(
                coordinates_iterable,
                total = len(longitudes),
                desc  = "Requesting seismograms"
            )
        
        times = None
        displacements_normal    = []
        displacements_longitude = []
        displacements_latitude  = []
        for longitude, latitude in coordinates_iterable:
            time, displacement_normal, displacement_longitude, displacement_latitude = self.request_seismogram(longitude, latitude)
            
            if times is None: times = time
            else:
                assert np.all(time == times), "Syngine returned different times vectors for different coordinates"
            
            displacements_normal.append(displacement_normal)
            displacements_longitude.append(displacement_longitude)
            displacements_latitude.append(displacement_latitude)

        displacements_normal    = np.array(displacements_normal)
        displacements_longitude = np.array(displacements_longitude)
        displacements_latitude  = np.array(displacements_latitude)

        return times, displacements_normal, displacements_longitude, displacements_latitude

    def request_fibre_path_seismograms(self, fibre: Fibre, verbose: bool = False):
        """
        Request seismograms from Syngine at fibre path segment coordinates.

        Inputs:
        - fibre: the fibre at which sections to request earthquakes. The fibre must be initialised with path coordinates.
        - verbose: whether to print requesting progress

        Outputs:
        - [np.ndarray] sample timestamps in s, length T
        - [np.ndarray] normal ground displacement over time in m, shape [P+1, T] with number of path segments S
        - [np.ndarray] longitude (east-west) ground displacement over time in m, shape [P+1, T]
        - [np.ndarray] latitude (north-south) ground displacement over time in m, shape [P+1, T]
        """
        return self.request_seismograms(*fibre.path_coordinates.T, verbose)

    def request_fibre_section_seismograms(self, fibre: Fibre, verbose: bool = False):
        """
        Request seismograms from Syngine at fibre section coordinates.

        Inputs:
        - fibre: the fibre at which sections to request earthquakes. The fibre must be initialised with path coordinates.
        - verbose: whether to print requesting progress

        Outputs:
        - [np.ndarray] sample timestamps in s, length T
        - [np.ndarray] normal ground displacement over time in m, shape [S+1, T] with number of fibre sections S
        - [np.ndarray] longitude (east-west) ground displacement over time in m, shape [S+1, T]
        - [np.ndarray] latitude (north-south) ground displacement over time in m, shape [S+1, T]
        """
        return self.request_seismograms(*fibre.section_coordinates.T, verbose)

    def request_fibre_section_projected_seismogram(self, fibre: Fibre, verbose: bool = False):
        """
        Request seismograms from Syngine at fibre section coordinates, projected onto the direction of each fibre section.

        Inputs:
        - fibre: the fibre at which sections to request earthquakes. The fibre must be initialised with path coordinates.
        - verbose: whether to print requesting progress

        Outputs:
        - [np.ndarray] sample timestamps in s, length T
        - [np.ndarray] normal ground displacement over time in m, shape [S+1, T] with number of fibre sections S
        - [np.ndarray] longitude (east-west) ground displacement over time in m, shape [S+1, T]
        - [np.ndarray] latitude (north-south) ground displacement over time in m, shape [S+1, T]
        - [np.ndarray] projected ground displacement over time in m, shape [S, T, 2]; the last dimension distinguishes between section ends
        """
        times, displacements_normal, displacements_longitude, displacements_latitude = self.request_fibre_section_seismograms(fibre, verbose)
        
        directions = np.diff(fibre.section_coordinates, axis = 0)
        displacements = np.block([
            [displacements_longitude[:-1, :, None, None], displacements_latitude[:-1, :, None, None]],
            [displacements_longitude[1:,  :, None, None], displacements_latitude[1:,  :, None, None]]
        ])

        displacements_projected = np.sum(displacements * directions[:, None, None], axis = 3) / np.linalg.norm(directions, axis = 1)[:, None, None]

        return times, displacements_normal, displacements_longitude, displacements_latitude, displacements_projected

    def request_fibre_section_projected_strain(self, fibre: Fibre, verbose: bool = False):
        """
        Request seismograms from Syngine at fibre section coordinates, projected onto the direction of each fibre section.
        Differentiate the projected seismogram to obtain material strain.

        Inputs:
        - fibre: the fibre at which sections to request earthquakes. The fibre must be initialised with path coordinates.
        - verbose: whether to print requesting progress

        Outputs:
        - [np.ndarray] sample timestamps in s, length T
        - [np.ndarray] normal ground displacement over time in m, shape [S+1, T] with number of fibre sections S
        - [np.ndarray] longitude (east-west) ground displacement over time in m, shape [S+1, T]
        - [np.ndarray] latitude (north-south) ground displacement over time in m, shape [S+1, T]
        - [np.ndarray] projected ground displacement over time in m, shape [S, T, 2]; the last dimension distinguishes between section ends
        - [np.ndarray] projected material strain over time, shape [S, T]
        """
        times, displacements_normal, displacements_longitude, displacements_latitude, displacements_projected = self.request_fibre_section_projected_seismogram(fibre, verbose)
        strains_projected = np.diff(displacements_projected, axis = 2)[:, :, 0] / fibre.section_lengths[:, None]

        return times, displacements_normal, displacements_longitude, displacements_latitude, displacements_projected, strains_projected

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