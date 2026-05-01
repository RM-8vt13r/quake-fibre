"""
Test correctness of earthquake.py
"""
from configparser import ConfigParser
import json

import numpy as np

from tremor_waveplate_toolbox import EarthquakeTerrestrial, EarthquakeSubmarine, Path

parameters = ConfigParser()

path_coordinates = [
    [102.57171090634661, 5.791616724837154],
    [102.72290646910318, 5.906566563564761],
    [102.75470997931428, 5.902883822383411]
]

parameters['EARTHQUAKE'] = {
    'event': 'GCMT:C201002270634A', # A historic earthquake event, structured <catalog>:<identifier> (e.g. from https://www.globalcmt.org/)
    'model': 'ak135f_5s',           # Earth model for Syngine to use from https://ds.iris.edu/ds/products/syngine/#earth
    'water_density': '1035',        # Water density at the seafloor in kg / m3
    'water_sound_velocity': '1510', # Speed of sound through water at the seafloor in m / s
    'water_depth': '2000',          # Depth of the sea in m
    'water_compressible': 'false',  # Whether the water column is compressible
    'strain_coefficient': '3.5e-9', # Coefficient to convert from hydrostatic pressure to fibre strain in 1 / Pa
    'batch_size_sparse': '6',       # The number of seismograms to request at most simultaneously
    'batch_size_dense': '50',       # The number of seismograms to request at most simultaneously
    'step_length_sparse': '1',      # Distances at which to request seismograms in km; these seismograms are interpolated to all fibre section 
    'step_length_dense': '0.1',     # Distances at which to request seismograms in km; these seismograms are interpolated to all fibre section 
    'worker_count': '5',            # Number of threads to request seismograms from Syngine simultaneously, high numbers (>20) might yield a temporary block
    'request_delay': '0.1',         # Minimum delay between filing concurrent seismogram requests (guidelines at https://ds.iris.edu/ds/nodes/dmc/services/usage/)
    'ray_resolution': '0.5'         # Step size in degrees to generate a ray parameter lookup table
}

def test_earthquakes():
    path = Path(*zip(*path_coordinates))
    
    earthquake_terrestrial = EarthquakeTerrestrial(parameters)
    # displacements_local_terrestrial, displacements_global, displacements_projected, strains_terrestrial = earthquake_terrestrial.request_fibre_strains(path, None, None, parameters.getint('EARTHQUAKE', 'batch_size_sparse'), parameters.getint('EARTHQUAKE', 'worker_count'), parameters.getfloat('EARTHQUAKE', 'request_delay'))
    earthquake_path, local_seismograms_terrestrial = earthquake_terrestrial.request_local_seismograms(path, None, None, parameters.getint('EARTHQUAKE', 'batch_size_sparse'), parameters.getint('EARTHQUAKE', 'worker_count'), parameters.getfloat('EARTHQUAKE', 'request_delay'))
    global_seismograms = earthquake_terrestrial.get_global_seismograms(local_seismograms_terrestrial, path, earthquake_path)
    projected_seismograms = earthquake_terrestrial.get_projected_seismograms(global_seismograms, path)
    strains_terrestrial = earthquake_terrestrial.get_fibre_strains(projected_seismograms, path)
    assert local_seismograms_terrestrial.shape[0] == path.vertex_count, f"path vertex count should match the first dimension of local_seismograms_terrestrial, but these were {path.vertex_count} and {local_seismograms_terrestrial.shape}"
    assert local_seismograms_terrestrial.shape[2] == 3, f"displacelocal_seismograms_terrestrialments_local_terrestrial should have three channels (normal, longitude, latitude), but had {local_seismograms_terrestrial.shape[2]}"
    assert global_seismograms.shape[0] == path.vertex_count, f"path vertex count should match the first dimension of global_seismograms, but these were {path.vertex_count} and {global_seismograms.shape[0]}"
    assert global_seismograms.shape[2] == 3, f"global_seismograms should have three channels (normal, longitude, latitude), but had {global_seismograms.shape[2]}"
    assert local_seismograms_terrestrial.shape == global_seismograms.shape, f"local_seismograms_terrestrial and global_seismograms should have the same shapes, but had shapes {local_seismograms_terrestrial.shape} and {global_seismograms.shape}"
    assert projected_seismograms.shape == (local_seismograms_terrestrial.shape[0] - 1, local_seismograms_terrestrial.shape[1], 2), f"projected_seismograms and local_seismograms_terrestrial should have shapes [S, T, 2] and [S + 1, T, 3], but had shapes {projected_seismograms.shape} and {local_seismograms_terrestrial.shape}"
    assert strains_terrestrial.shape == (*projected_seismograms.shape[:-1], 1), f"strains_terrestrial and projected_seismograms should have shapes [S, T, 1] and [S, T, 2], but had shapes {strains_terrestrial.shape} and {projected_seismograms.shape}"

    earthquake_submarine = EarthquakeSubmarine(parameters)
    # displacements_local_submarine, normal_accelerations, differential_pressures, strains_submarine = earthquake_submarine.request_fibre_strains(path, None, None, parameters.getint('EARTHQUAKE', 'batch_size_sparse'), parameters.getint('EARTHQUAKE', 'worker_count'), parameters.getfloat('EARTHQUAKE', 'request_delay'))
    earthquake_path, local_seismograms_submarine = earthquake_submarine.request_local_seismograms(path, None, None, parameters.getint('EARTHQUAKE', 'batch_size_sparse'), parameters.getint('EARTHQUAKE', 'worker_count'), parameters.getfloat('EARTHQUAKE', 'request_delay'))
    normal_accelerations = earthquake_submarine.get_normal_accelerations(local_seismograms_submarine, path, earthquake_path)
    differential_pressures = earthquake_submarine.get_differential_pressures(normal_accelerations, path)
    strains_submarine = earthquake_submarine.get_fibre_strains(differential_pressures)
    assert local_seismograms_submarine.shape[0] == path.edge_count, f"path edge count should match the first dimension of local_seismograms_submarine, but these were {path.edge_count} and {local_seismograms_submarine.shape}"
    assert local_seismograms_submarine.shape[2] == 3, f"local_seismograms_submarine should have three channels (normal, longitude, latitude), but had {local_seismograms_submarine.shape[2]}"
    assert normal_accelerations.shape[0] == path.edge_count, f"path edge count should match the first dimension of normal_accelerations, but these were {path.edge_count} and {normal_accelerations.shape[0]}"
    assert normal_accelerations.shape[2] == 1, f"normal_accelerations should have one channel (normal), but had {normal_accelerations.shape[2]}"
    assert local_seismograms_submarine.shape[:2] == normal_accelerations.shape[:2], f"local_seismograms_submarine and normal_accelerations should have shapes [S, T, 3] and [S, T, 1], but had shapes {local_seismograms_submarine.shape} and {normal_accelerations.shape}"
    assert differential_pressures.shape == normal_accelerations.shape, f"differential_pressures and normal_accelerations should have the same shapes, but had shapes {differential_pressures.shape} and {normal_accelerations.shape}"
    assert strains_submarine.shape == differential_pressures.shape, f"strains_submarine and differential_pressures should have the same shapes, but had shapes {strains_submarine.shape} and {differential_pressures.shape}"

    assert not np.allclose(strains_submarine.samples_time, strains_terrestrial.samples_time), f"Strains in terrestrial and submarine fibres were the same for the same earthquake, but should be different."

def test_concurrency():
    path = Path(*zip(*path_coordinates))
    earthquake = EarthquakeTerrestrial(parameters)

    fibre_strains_sequential = earthquake.request_fibre_strains(path, None, None, parameters.getint('EARTHQUAKE', 'batch_size_sparse'), 1, 0)
    fibre_strains_concurrent = earthquake.request_fibre_strains(path, None, None, 1, parameters.getint('EARTHQUAKE', 'worker_count'), parameters.getfloat('EARTHQUAKE', 'request_delay'))
    
    assert np.allclose(fibre_strains_sequential.samples_time, fibre_strains_concurrent.samples_time), f"Earthquake strains must match when requested sequentially or concurrently, but didn't"

def test_interpolation():
    path = Path(*zip(*path_coordinates)).interpolated(parameters.getfloat('EARTHQUAKE', 'step_length_dense'))

    # earthquake_terrestrial = EarthquakeTerrestrial(parameters)
    # strains_terrestrial_interpolated = earthquake_terrestrial.request_fibre_strains(
    #         path,
    #         parameters.getfloat('EARTHQUAKE', 'step_length_sparse'),
    #         None,
    #         parameters.getint('EARTHQUAKE', 'batch_size_sparse'),
    #         parameters.getint('EARTHQUAKE', 'worker_count'),
    #         parameters.getfloat('EARTHQUAKE', 'request_delay')
    #     )

    # strains_terrestrial_raw = earthquake_terrestrial.request_fibre_strains(
    #         path,
    #         None,
    #         None,
    #         parameters.getint('EARTHQUAKE', 'batch_size_dense'),
    #         parameters.getint('EARTHQUAKE', 'worker_count'),
    #         parameters.getfloat('EARTHQUAKE', 'request_delay')
    #     )

    # assert strains_terrestrial_interpolated.shape == strains_terrestrial_raw.shape, f"Synthesised and interpolated terrestrial earthquake strains must have the same shapes, but these were {strains_terrestrial_raw.shape} and {strains_terrestrial_interpolated.shape}"
    # for time_index in range(strains_terrestrial_raw.shape[1]):
    #     assert np.allclose(
    #             np.interp(path.interpolated(parameters.getfloat('EARTHQUAKE', 'step_length_sparse')).centre_positions, path.centre_positions, strains_terrestrial_raw.samples_time[:, time_index, 0]),
    #             np.interp(path.interpolated(parameters.getfloat('EARTHQUAKE', 'step_length_sparse')).centre_positions, path.centre_positions, strains_terrestrial_interpolated.samples_time[:, time_index, 0]),
    #             atol = 0, rtol = 0.1
    #         ), "Synthesised and interpolated terrestrial seismogram values must match, but didn't"
    
    earthquake_submarine = EarthquakeSubmarine(parameters)
    strains_submarine_interpolated = earthquake_submarine.request_fibre_strains(
            path,
            parameters.getfloat('EARTHQUAKE', 'step_length_sparse'),
            None,
            parameters.getint('EARTHQUAKE', 'batch_size_sparse'),
            parameters.getint('EARTHQUAKE', 'worker_count'),
            parameters.getfloat('EARTHQUAKE', 'request_delay')
        )

    strains_submarine_raw = earthquake_submarine.request_fibre_strains(
            path,
            None,
            None,
            parameters.getint('EARTHQUAKE', 'batch_size_dense'),
            parameters.getint('EARTHQUAKE', 'worker_count'),
            parameters.getfloat('EARTHQUAKE', 'request_delay')
        )

    assert strains_submarine_interpolated.shape == strains_submarine_raw.shape, f"Synthesised and interpolated submarine earthquake strains must have the same shapes, but these were {strains_submarine_raw.shape} and {strains_submarine_interpolated.shape}"
    for time_index in range(strains_submarine_raw.shape[1]):
        assert np.allclose(
                np.interp(path.interpolated(parameters.getfloat('EARTHQUAKE', 'step_length_sparse')).centre_positions, path.centre_positions, strains_submarine_raw.samples_time[:, time_index, 0])[1:], # [1:] to disregard boundary conditions
                np.interp(path.interpolated(parameters.getfloat('EARTHQUAKE', 'step_length_sparse')).centre_positions, path.centre_positions, strains_submarine_interpolated.samples_time[:, time_index, 0])[1:],
                atol = np.max(np.abs(strains_submarine_raw.samples_time[:, time_index, 0])) / 20,
                rtol = 0.1
            ), "Synthesised and interpolated submarine seismogram values must match, but didn't"

def test_intervals():
    path = Path(*zip(*path_coordinates)).interpolated(parameters.getfloat('EARTHQUAKE', 'step_length_dense'))
    earthquake = EarthquakeTerrestrial(parameters)

    perturbation = earthquake(
            path,
            None,
            1,
            parameters.getint('EARTHQUAKE', 'batch_size_dense'),
            parameters.getint('EARTHQUAKE', 'worker_count'),
            parameters.getfloat('EARTHQUAKE', 'request_delay')
        )

    assert perturbation.duration == 1, f"Requested perturbation for 1 second, but it lasted {perturbation.duration}"
    assert perturbation.start_time == 0, f"Perturbation should start at 0s, but started at {perturbation.start_time} s"