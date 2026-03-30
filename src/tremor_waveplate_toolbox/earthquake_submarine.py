"""
A class that simulates earthquakes on the sea floor along a path on the earth using Syngine.
"""
from configparser import ConfigParser
import logging
from typing import override

import numpy as np
import obspy as op
import obspy.taup

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
        assert 'water_compressible'   in parameters['EARTHQUAKE'], "'compressible' is missing from parameters section 'EARTHQUAKE'"
        assert 'strain_coefficient'   in parameters['EARTHQUAKE'], "'strain_coefficient' is missing from parameters section 'EARTHQUAKE'."
        assert 'ray_resolution'       in parameters['EARTHQUAKE'], "'ray_resolution' is missing from parameters section 'EARTHQUAKE'"
        
        self._water_sound_velocity = parameters.getfloat('EARTHQUAKE', 'water_sound_velocity') # Speed of sound through water at the ocean floor in m / s
        self._water_density        = parameters.getfloat('EARTHQUAKE', 'water_density') # Water density at the seafloor in kg / m3
        self._water_depth          = parameters.getfloat('EARTHQUAKE', 'water_depth')   # Depth of the sea in m
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

        if EarthquakeSubmarine.RAY_PARAMETERS is None or EarthquakeSubmarine.RAY_PARAMETERS[1] - EarthquakeSubmarine.RAY_PARAMETERS[0] > ray_resolution:
            logger.info(f"Initialising ray parameters table with resolution {ray_resolution} degrees..")

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
                earthquake_path: Path,
                batch_size: int
            ) -> (int, list[np.ndarray], np.ndarray, np.ndarray):
        """
        Part of request_local_seismograms() that builds batches of HTTP requests to send to Syngine.
        The EarthquakeSubmarine version puts batch coordinates in the centre of each fibre section.
        """
        return super()._local_seismograms_build_batches(earthquake_path.centre_coordinates, batch_size)

    @override
    def _local_seismograms_postprocess(self, earthquake_path: Path, syngine_stream: list):
        return super()._local_seismograms_postprocess(earthquake_path.edge_count, syngine_stream)

    def _normal_displacements_interpolate(self, earthquake_path: Path, path: Path, normal_displacements: np.ndarray):
        """
        Part of request_normal_accelerations() that interpolates sparsely obtained normal displacements to a denser path.
        """
        normal_displacements_interpolated_flattened = np.zeros(shape = (normal_displacements.shape[-2], path.edge_count)) # [T, C]
        normal_displacements_flattened = normal_displacements[:, :, 0].transpose() # [I, T, 1] -> [T, I]

        for channel_index, normal_displacement_flattened in enumerate(normal_displacements_flattened):
            normal_displacements_interpolated_flattened[channel_index] = np.interp(path.centre_positions, earthquake_path.centre_positions, normal_displacement_flattened)

        normal_displacements_interpolated = normal_displacements_interpolated_flattened.transpose()[:, :, None] # [T, C] -> [C, T, 1]

        logger.info(f"Normal displacements interpolated from {earthquake_path.edge_count} sections to {path.edge_count} sections")

        return normal_displacements_interpolated

    def request_normal_accelerations(self,
                path: Path,
                step_length: float = None,
                duration: float = None,
                return_displacements_local: bool = False,
                batch_size: int = None,
                worker_count: int = 1,
                request_delay: float = 0.1
            ):
        """
        Request seismograms from Syngine at fibre section centres, and transform them to normal seafloor acceleration at each coordinate.

        Inputs:
        - path [Path]: Fibre path with C edges
        - step_length [float]: if not None, request earthquakes at I points along path spaced step_length apart in km.
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - return_displacements_local [bool]: if True, return displacements in local and global coordinates.
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C
        - worker_count [int]: how many Syngine requests to make at most in parallel
        - request_delay [float]: minimum delay in seconds between launching two Syngine requests
        
        Outputs:
        - If return_displacements_local: [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [I, T, D] where D indexes longitudinal, latitudinal, and normal components in that order
        - [Signal] signal containing normal seafloor acceleration in m / s2, shape [C, T, 1].
        """
        earthquake_path, displacements_local = self.request_local_seismograms(path, step_length, duration, batch_size, worker_count, request_delay)

        normal_displacements = np.zeros(shape = (path.edge_count, displacements_local.shape[1] + 2, 1))
        if step_length is not None:
            normal_displacements[:, 1:-1] = self._normal_displacements_interpolate(earthquake_path, path, displacements_local.samples_time[:, :, 2, None])
        else:
            normal_displacements[:, 1:-1] = displacements_local.samples_time[:, :, 2, None]

        normal_accelerations = Signal(
            samples = (normal_displacements[:, :-2] - 2 * normal_displacements[:, 1:-1] + normal_displacements[:, 2:]) * displacements_local.sample_rate ** 2,
            sample_rate = displacements_local.sample_rate
        )

        return_list = []
        if return_displacements_local: return_list.append(displacements_local)
        return_list.append(normal_accelerations)

        logger.debug("Returning normal accelerations")
        return return_list

    def request_incompressible_differential_pressures(self,
                path: Path,
                step_length: float = None,
                duration: float = None,
                filter_frequencies: np.ndarray = None,
                filter_taps: np.ndarray = None,
                return_displacements_local: bool = False,
                return_normal_accelerations: bool = False,
                batch_size: int = None,
                worker_count: int = 1,
                request_delay: float = 0.1
            ):
        """
        Interpreting longitudes and latitudes as chronological path coordinates, obtain differential seafloor water pressure at each edge center of this path, assuming an incompressible water column.

        Inputs:
        - path [Path]: Fibre path with C edges
        - step_length [float]: if not None, request earthquakes at I points along path spaced step_length apart in km.
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - filter_frequencies [np.ndarray]: temporary toolbox function. If not None, define a frequency-domain filter with which to filter obtained differential pressures.
        - filter_taps [np.ndarray]: temporary toolbox function. If not None, define a frequency-domain filter with which to filter obtained differential pressures.
        - return_displacements_local [bool]: if True, return displacements in local and global coordinates.
        - return_normal_accelerations [bool]: if True, return acceleration of the seafloor away from the earth's centre.
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C
        - worker_count [int]: how many Syngine requests to make at most in parallel
        - request_delay [float]: minimum delay in seconds between launching two Syngine requests
        
        Outputs:
        - If return_displacements_local: [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [I, T, D] where D indexes longitudinal, latitudinal, and normal components in that order
        - If return_normal_accelerations: [Signal] signal containing normal seafloor acceleration in m / s2, shape [C, T, 1]
        - [Signal] signal containing differential water pressure at the seafloor in Pa under the assumption of an incompressible water column, shape [C, T, 1].
        """
        if filter_frequencies is not None or filter_taps is not None:
            assert filter_frequencies is not None and filter_taps is not None, f"filter_frequencies and filter_taps must both be None or not None, but were {filter_frequencies} and {filter_taps}"

        results_list = self.request_normal_accelerations(path, step_length, duration, return_displacements_local, batch_size, worker_count, request_delay)
        normal_accelerations = results_list[-1]
        
        incompressible_differential_pressures = Signal(
            samples = self.water_density * self.water_depth * normal_accelerations.samples_time,
            sample_rate = normal_accelerations.sample_rate
        )

        if filter_frequencies is not None:
            incompressible_differential_pressures.samples_frequency *= np.interp(incompressible_differential_pressures.frequency % incompressible_differential_pressures.bandwidth, filter_frequencies, filter_taps)[None, :, None]
        
        return_list = results_list[:-1]
        if return_normal_accelerations: return_list.append(normal_accelerations)
        return_list.append(incompressible_differential_pressures)

        logger.debug("Returning incompressible differential pressures")
        return return_list

    def request_differential_pressures(self,
                path: Path,
                step_length: float = None,
                duration: float = None,
                filter_frequencies: np.ndarray = None,
                filter_taps: np.ndarray = None,
                return_displacements_local: bool = False,
                return_normal_accelerations: bool = False,
                return_incompressible_differential_pressures: bool = False,
                batch_size: int = None,
                worker_count: int = 1,
                request_delay: float = 0.1
            ):
        """
        Interpreting longitudes and latitudes as chronological path coordinates, obtain differential seafloor water pressure at each edge center of this path.

        Inputs:
        - path [Path]: Fibre path with C edges
        - step_length [float]: if not None, request earthquakes at I points along path spaced step_length apart in km.
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - filter_frequencies [np.ndarray]: temporary toolbox function. If not None, define a frequency-domain filter with which to filter obtained differential pressures.
        - filter_taps [np.ndarray]: temporary toolbox function. If not None, define a frequency-domain filter with which to filter obtained differential pressures.
        - return_displacements_local [bool]: if True, return displacements in local and global coordinates.
        - return_normal_accelerations [bool]: if True, return acceleration of the seafloor away from the earth's centre.
        - return_incompressible_differential_pressures [bool]: if True, return differential pressures at the seafloor when the seawater is incompressible.
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C
        - worker_count [int]: how many Syngine requests to make at most in parallel
        - request_delay [float]: minimum delay in seconds between launching two Syngine requests
        
        Outputs:
        - If return_displacements_local: [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [I, T, D] where D indexes longitudinal, latitudinal, and normal components in that order
        - If return_normal_accelerations: [Signal] signal containing normal seafloor acceleration in m / s2, shape [C, T, 1]
        - If return_incompressible_differential_pressures: [Signal] signal containing differential water pressure at the seafloor in Pa under the assumption of an incompressible water column, shape [C, T, 1].
        - [Signal] signal containing differential water pressure at the seafloor Pa, shape [C, T, 1].
        """
        results_list = self.request_incompressible_differential_pressures(path, step_length, duration, return_displacements_local, batch_size, worker_count, request_delay)
        incompressible_differential_pressures = results_list[-1]

        # Obtain ray parameter for each path vertex
        distance_angles = op.geodetics.base.locations2degrees(path.centre_latitudes, path.centre_longitudes, self.origin.latitude, self.origin.longitude)
        ray_parameters  = np.interp(distance_angles, EarthquakeSubmarine.RAY_ANGLES, EarthquakeSubmarine.RAY_PARAMETERS)

        constants = np.sqrt(1 - ray_parameters ** 2 * self.water_sound_velocity ** 2)
        differential_pressures = incompressible_differential_pressures.copy()
        np.divide(
            differential_pressures.samples_frequency * self.water_sound_velocity * np.tan(normal_accelerations.frequency_angular[None, :, None] * self.water_depth * constants[:, None, None] / self.water_sound_velocity),
            normal_accelerations.frequency_angular[None, :, None] * self.water_depth * constants[:, None, None],
            out = differential_pressures.samples_frequency,
            where = normal_accelerations.frequency_angular[None, :, None] != 0
        )
        
        return_list = results_list[:-1]
        if return_incompressible_differential_pressures: return_list.append(incompressible_differential_pressures)
        return_list.append(differential_pressures)

        logger.debug("Returning differential pressures")
        return return_list

    def request_fibre_strains(self,
                path: Path,
                step_length: float = None,
                duration: float = None,
                filter_frequencies: np.ndarray = None,
                filter_taps: np.ndarray = None,
                return_displacements_local: bool = False,
                return_normal_accelerations: bool = False,
                return_incompressible_differential_pressures: bool = False,
                return_differential_pressures: bool = False,
                batch_size: int = None,
                worker_count: int = 1,
                request_delay: float = 0.1
            ):
        """
        Interpreting longitudes and latitudes as chronological path coordinates, request the longitudinal material strain on each path section by scaling the water pressure differences.

        Inputs:
        - path [Path]: Fibre path with C edges
        - step_length [float]: if not None, request earthquakes at I points along path spaced step_length apart in km. Then, interpolate the results back to every edge centre along path to calculate strains.
        - duration [float]: duration from the earthquake origin for which to synthesize seismograms. If None, synthesize the whole event.
        - filter_frequencies [np.ndarray]: temporary toolbox function. If not None, define a frequency-domain filter with which to filter obtained differential pressures.
        - filter_taps [np.ndarray]: temporary toolbox function. If not None, define a frequency-domain filter with which to filter obtained differential pressures.
        - return_displacements_local [bool]: if True, return displacements in local and global coordinates.
        - return_normal_accelerations [bool]: if True, return acceleration of the seafloor away from the earth's centre.
        - return_differential_pressures [bool]: if True, return differential water pressure at the sea floor.
        - return_incompressible_differential_pressures [bool]: if True, return differential pressures at the seafloor when the seawater is incompressible.
        - batch_size [int]: how many seismograms to request simultaneously; defaults to C
        - worker_count [int]: how many Syngine requests to make at most in parallel
        - request_delay [float]: minimum delay in seconds between launching two Syngine requests

        Outputs:
        - If return_displacements_local: [Signal] signal containing all three displacement components in m, relative to local coordinates, shape [I, T, D] where D indexes longitudinal, latitudinal, and normal components in that order
        - If return_normal_accelerations: [Signal] signal containing normal seafloor acceleration in m / s2, shape [C, T, 1]
        - If return_incompressible_differential_pressures: [Signal] signal containing differential water pressure at the seafloor in Pa under the assumption of an incompressible water column, shape [C, T, 1].
        - If return_differential_pressures: [Signal] signal containing differential water pressure at the seafloor Pa, shape [C, T, 1].
        - [Signal] signal containing fibre strain, shape [C, T, 1].
        """
        if self.water_compressible:
            results_list = self.request_differential_pressures(path, step_length, duration, filter_frequencies, filter_taps, return_displacements_local, return_normal_accelerations, return_incompressible_differential_pressures, batch_size, worker_count, request_delay)
        else:
            assert return_differential_pressures == False, f"return_differential_pressures must be false for earthquake with no compressible water column"
            results_list = self.request_incompressible_differential_pressures(path, step_length, duration, filter_frequencies, filter_taps, return_displacements_local, return_normal_accelerations, batch_size, worker_count, request_delay)
        
        differential_pressures = results_list[-1]

        fibre_strains = Signal(
            samples = self.strain_coefficient * differential_pressures.samples_time,
            sample_rate = differential_pressures.sample_rate
        )

        return_list = results_list[:-1]
        if return_differential_pressures: return_list.append(differential_pressures)
        return_list.append(fibre_strains)

        logger.debug("Returning fibre strains")
        return return_list

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
        [float] The depth of the sea in m, assumed constant
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