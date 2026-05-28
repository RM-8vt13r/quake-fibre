"""
A class that simulates earthquakes on the sea floor along a path on the earth using Syngine.
"""
from configparser import ConfigParser
import logging
from typing import override

import numpy as np
import obspy as op
import obspy.taup
import netCDF4 as nc

from .earthquake import Earthquake
from .signal import Signal
from .path import Path

logger = logging.getLogger()

class EarthquakeSubmarine(Earthquake):
    RAY_ANGLES = None
    RAY_PARAMETERS = None

    @override
    def __init__(self, parameters: ConfigParser):
        """
        Initialise the submarine earthquake

        Required entries in parameters['EARTHQUAKE'], in addition to those listed in the Earthquake class:
        - water_sound_velocity [float]: Speed of sound through water at the ocean floor in m / s
        - water_density [float]: Water density at the seafloor in kg / m3
        - water_depth [float]: Depth of the sea in m
        - water_compressible [float]: Whether the water column is assumed compressible or not
        - strain_coefficient [float]: Coupling coefficient from pressure to strain in 1 / Pa
        - ray_resolution [float]: EarthquakeSubmarine generates a lookup table upon creation; ray_resolution is the angle step size for this table in degrees
        """
        super().__init__(parameters)

        assert 'water_sound_velocity' in parameters['EARTHQUAKE'], "'water_sound_velocity' is missing from parameters section 'EARTHQUAKE'."
        assert 'water_density'        in parameters['EARTHQUAKE'], "'water_density' is missing from parameters section 'EARTHQUAKE'."
        assert 'water_depth'          in parameters['EARTHQUAKE'], "'water_depth' is missing from parameters section 'EARTHQUAKE'."
        assert 'water_compressible'   in parameters['EARTHQUAKE'], "'water_compressible' is missing from parameters section 'EARTHQUAKE'"
        assert 'strain_coefficient'   in parameters['EARTHQUAKE'], "'strain_coefficient' is missing from parameters section 'EARTHQUAKE'."
        assert 'ray_resolution'       in parameters['EARTHQUAKE'], "'ray_resolution' is missing from parameters section 'EARTHQUAKE'"
        
        self._water_sound_velocity = parameters.getfloat('EARTHQUAKE', 'water_sound_velocity') # Speed of sound through water at the ocean floor in m / s
        self._water_density        = parameters.getfloat('EARTHQUAKE', 'water_density') # Water density at the seafloor in kg / m3
        try:
            self._water_depth      = parameters.getfloat('EARTHQUAKE', 'water_depth')   # Depth of the sea in m
        except:
            self._water_depth      = parameters.get('EARTHQUAKE', 'water_depth') # DAP link to bathymetric map
        self._water_compressible   = parameters.getboolean('EARTHQUAKE', 'water_compressible') # Whether the water column is compressible or not
        self._strain_coefficient   = parameters.getfloat('EARTHQUAKE', 'strain_coefficient') # Strain coefficient in 1 / Pa
        ray_resolution             = parameters.getfloat('EARTHQUAKE', 'ray_resolution') # Step size in degrees to generate the ray parameter lookup table with

        assert ray_resolution > 0 and ray_resolution <= 180, f"ray_resolution must be between 0 and 180, but was {ray_resolution}"

        for field in (
                '_water_sound_velocity',
                '_water_density',
                '_water_depth'
            ):
            assert getattr(self, field) > 0, f"{field} must be >0, but was {field}"

        if (EarthquakeSubmarine.RAY_PARAMETERS is None or EarthquakeSubmarine.RAY_PARAMETERS[1] - EarthquakeSubmarine.RAY_PARAMETERS[0] > ray_resolution) and self.water_compressible:
            logger.info(f"Initialising ray parameters table with resolution {ray_resolution} degrees")

            EarthquakeSubmarine.RAY_ANGLES = np.append(np.arange(0, 180, ray_resolution), 180)
            
            model = self.model.split('_')[0].lower()
            taup_model = op.taup.TauPyModel(model = model if model != 'ak135f' else 'ak135')
            EarthquakeSubmarine.RAY_PARAMETERS = []
            for angle in EarthquakeSubmarine.RAY_ANGLES:
                travel_times = taup_model.get_travel_times(
                        source_depth_in_km = self.origin.depth / 1000,
                        distance_in_degree = angle,
                    )
                travel_time = min(travel_times, key = lambda x: x.time)
                ray_parameter = travel_time.ray_param_sec_degree / op.geodetics.degrees2kilometers(1000) # s / m
                EarthquakeSubmarine.RAY_PARAMETERS.append(ray_parameter)

            EarthquakeSubmarine.RAY_PARAMETERS = np.array(EarthquakeSubmarine.RAY_PARAMETERS)

    @override
    def _local_seismograms_build_batches(self,
                path: Path,
                batch_size: int
            ) -> (int, list[np.ndarray], np.ndarray, np.ndarray):
        """
        Part of request_local_seismograms() that builds batches of HTTP requests to send to Syngine.
        The EarthquakeSubmarine version puts batch coordinates in the centre of each fibre section.
        """
        return super()._local_seismograms_build_batches(path.centre_coordinates, batch_size)

    @override
    def _local_seismograms_postprocess(self,
                path: Path,
                syngine_stream: list
            ):
        return super()._local_seismograms_postprocess(path.edge_count, syngine_stream)

    def get_normal_accelerations(self,
                local_seismograms: Signal,
                path: Path
            ) -> Signal:
        """
        Request seismograms from Syngine at fibre section centres, and transform them to normal seafloor acceleration at each coordinate.

        Inputs:
        - local_seismograms [Signal]: signal containing all three displacement components in m, relative to local coordinates, shape [I, T, D] where D indexes longitudinal, latitudinal, and normal components in that order
        - path [Path]: Fibre path with C edges
        
        Outputs:
        - [Signal] signal containing normal seafloor acceleration in m / s2, shape [C, T, 1].
        """
        normal_displacements = np.zeros(shape = (path.edge_count, local_seismograms.shape[1] + 2, 1))
        normal_displacements[:, 1:-1] = local_seismograms.samples_time[:, :, 2, None]
        normal_accelerations = Signal(
            samples = (normal_displacements[:, :-2] - 2 * normal_displacements[:, 1:-1] + normal_displacements[:, 2:]) * local_seismograms.sample_rate ** 2,
            sample_rate = local_seismograms.sample_rate
        )

        # if earthquake_path != path:
        #     normal_accelerations.samples_time = self._normal_accelerations_interpolate(earthquake_path, path, normal_accelerations.samples_time)

        logger.debug("Returning normal accelerations")
        return normal_accelerations

    def request_water_depths(self,
                path: Path
            ) -> Signal:
        """
        Interpreting longitudes and latitudes as chronological path coordinates, obtain seafloor depths at each edge center of this path.
        The depth is either constant along the path if water_depth was defined, or loaded from a bathymetric map otherwise.

        Inputs:
        - path [Path]: Fibre path with C edges
        
        Outputs:
        - [Signal] seafloor depths in m, shape [C, 1, 1].
        """
        if not isinstance(self.water_depth, str):
            return Signal(np.full((*path.lengths.shape, 1, 1), self.water_depth), 1)

        water_depths = np.full(path.lengths.shape, None, dtype = float)
        with nc.Dataset(self.water_depth, 'r') as bathymetric_map:
            for index, (latitude, longitude) in enumerate(path.centre_coordinates):
                water_depths[index] = bathymetric_map.variables['elevation'][latitude, longitude]

        return Signal(water_depths[:, None, None], 1)

    def get_differential_pressures(self,
                normal_accelerations: Signal,
                water_depths: Signal,
                path: Path
            ) -> Signal:
        """
        Interpreting longitudes and latitudes as chronological path coordinates, obtain differential seafloor water pressure at each edge center of this path.

        Inputs:
        - normal_accelerations [Signal]: signal containing normal seafloor acceleration in m / s2, shape [C, T, 1]
        - water_depths [Signal]: depth of each path edge, shape [C, 1, 1]
        - path [Path]: Fibre path with C edges
        
        Outputs:
        - [Signal] signal containing differential water pressure at the seafloor Pa, shape [C, T, 1].
        """
        differential_pressures = Signal(
            samples = self.water_density * water_depths.samples_time * normal_accelerations.samples_time,
            sample_rate = normal_accelerations.sample_rate
        )

        if self.water_compressible:
            distance_angles = op.geodetics.base.locations2degrees(path.centre_latitudes, path.centre_longitudes, self.origin.latitude, self.origin.longitude)
            ray_parameters  = np.interp(distance_angles, EarthquakeSubmarine.RAY_ANGLES, EarthquakeSubmarine.RAY_PARAMETERS)

            constants = np.sqrt(1 - ray_parameters ** 2 * self.water_sound_velocity ** 2)[:, None, None]
            np.divide(
                differential_pressures.samples_frequency * self.water_sound_velocity * np.tan(normal_accelerations.frequency_angular[None, :, None] * water_depths.samples_time * constants / self.water_sound_velocity),
                normal_accelerations.frequency_angular[None, :, None] * water_depths.samples_time * constants,
                out = differential_pressures.samples_frequency,
                where = normal_accelerations.frequency_angular[None, :, None] != 0
            )
        
        logger.debug("Returning differential pressures")
        return differential_pressures

    def get_fibre_strains(self,
                differential_pressures: Signal
            ) -> Signal:
        """
        Calculate longitudinal material strain by scaling the water pressure differences.

        Inputs:
        - differential_pressures: [Signal] signal containing differential water pressure at the seafloor Pa, shape [C, T, 1].

        Outputs:
        - [Signal] signal containing fibre strain, shape [C, T, 1].
        """
        fibre_strains = Signal(
            samples = self.strain_coefficient * differential_pressures.samples_time,
            sample_rate = differential_pressures.sample_rate
        )

        logger.debug("Returning fibre strains")
        return fibre_strains

    @override
    def request_fibre_strains(self,
            path: Path,
            duration: float,
            batch_size: int,
            worker_count: int,
            request_delay: float,
            local_seismograms: Signal = None,
            normal_accelerations: Signal = None,
            water_depths: Signal = None,
            differential_pressures: Signal = None,
            return_local_seismograms: bool = False,
            return_normal_accelerations: bool = False,
            return_water_depths: bool = False,
            return_differential_pressures: bool = False
        ) -> Signal:
        """
        Interpreting longitudes and latitudes as chronological path coordinates, request the longitudinal material strain on each path section from Syngine.

        Inputs:
        - path [Path]: coordinates, length C
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C.
        - worker_count [int]: how many Syngine requests to make at most in parallel.
        - request_delay [float]: minimum delay in seconds between launching two Syngine requests.
        - local_seismograms [Signal]: if not None, use these local seismograms instead of requesting new ones, shape [C, T, D].
        - normal_accelerations [Signal]: if not None, use these normal accelerations instead of requesting new ones, shape [C, T, 1].
        - water_depths [Signal]: if not None, use these water depths instead of requesting new ones, shape [C, 1, 1].
        - differential_pressures [Signal]: if not None, use these differential pressures instead of requesting new ones, shape [C, T, 1]
        - return_local_seismograms [bool]: if True, return the obtained local seismograms in m.
        - return_normal_accelerations [bool]: if True, return the obtained normal accelerations in m/s2.
        - return_water_depths [bool]: if True, return the obtained water depths in m.
        - return_differential_pressures [bool]: if True, return the obtained differential pressures in Pa.

        Outputs:
        - [list or Signal] List of returned Signals in the following order:
          1. [Signal] if return_local_seismograms is True, the obtained local seismograms in m, shape [C, T, D] where T is the time axis and D indexes longitudinal, latitudinal, and normal components in that order.
          2. [Signal] if return_normal_accelerations is True, the obtained normal accelerations in m/s2, shape [C, T, 1].
          3. [Signal] if return_water_depths is True, the obtained water depths in m, shape [C, 1, 1].
          4. [Signal] if return_differential_pressures is True, the obtained differential pressures in Pa, shape [C, T, 1].
          5. [Signal] the obtained fibre strains, shape [C, T, 1].
          Or, if all return_* parameters are False, just a Signal with the obtained fibre strains
        """
        return_list = []

        if local_seismograms is None:
            local_seismograms = self.request_local_seismograms(path, duration, batch_size, worker_count, request_delay)

        if normal_accelerations is None:
            normal_accelerations = self.get_normal_accelerations(local_seismograms, path)
        if return_local_seismograms:
            return_list.append(local_seismograms)
        else:
            del local_seismograms

        if water_depths is None:
            water_depths = self.request_water_depths(path)

        if differential_pressures is None:
            differential_pressures = self.get_differential_pressures(normal_accelerations, water_depths, path)
        if return_normal_accelerations:
            return_list.append(normal_accelerations)
        else:
            del normal_accelerations
        if return_water_depths:
            return_list.append(water_depths)
        else:
            del water_depths

        fibre_strains = self.get_fibre_strains(differential_pressures)
        if return_differential_pressures:
            return_list.append(differential_pressures)
        else:
            del differential_pressures

        return_list.append(fibre_strains)
        return return_list if len(return_list) > 1 else return_list[0]

    @property
    def water_sound_velocity(self):
        """
        [float] The speed of sound in water at the seafloor, in m / s
        """
        return self._water_sound_velocity

    @water_sound_velocity.setter
    def water_sound_velocity(self, value):
        raise AttributeError("Cannot change water sound velocity after instantiation; create a new instance instead")

    @property
    def water_depth(self):
        """
        [float] The depth of the sea in m, assumed constant, OR, [str] the DAP link to a bathymetric map.
        """
        return self._water_depth

    @water_depth.setter
    def water_depth(self, value):
        raise AttributeError("Cannot change water depth after instantiation; create a new instance instead")

    @property
    def water_density(self):
        """
        [float] The density of water at the seafloor, in kg / m3
        """
        return self._water_density

    @water_density.setter
    def water_density(self, value):
        raise AttributeError("Cannot change water density after instantiation; create a new instance instead")

    @property
    def water_compressible(self):
        """
        [float] Whether the water column is vertically compressible
        """
        return self._water_compressible

    @water_compressible.setter
    def water_compressible(self, value):
        raise AttributeError("Cannot change water compressibility after instantiation; create a new instance instead")

    @property
    def strain_coefficient(self):
        """
        [float] The coefficient to translate from differential water pressure to fibre strain, in 1 / Pa
        """
        return self._strain_coefficient

    @strain_coefficient.setter
    def strain_coefficient(self, value):
        raise AttributeError("Cannot change strain coefficient after instantiation; create a new instance instead")