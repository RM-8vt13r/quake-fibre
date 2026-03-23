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
    'strain_coefficient': '3.5e-9', # Coefficient to convert from hydrostatic pressure to fibre strain in 1 / Pa
    'batch_size_sparse': '6',       # The number of seismograms to request at most simultaneously
    'batch_size_dense': '50',       # The number of seismograms to request at most simultaneously
    'step_length_sparse': '1',      # Distances at which to request seismograms in km; these seismograms are interpolated to all fibre section 
    'step_length_dense': '0.1',     # Distances at which to request seismograms in km; these seismograms are interpolated to all fibre section 
    'worker_count': '5',            # Number of threads to request seismograms from Syngine simultaneously, high numbers (>20) might yield a temporary block
    'request_delay': '0.1',         # Minimum delay between filing concurrent seismogram requests (guidelines at https://ds.iris.edu/ds/nodes/dmc/services/usage/)
    'ray_resolution': '0.5',        # Step size in degrees to generate a ray parameter lookup table
}

def test_earthquakes():
    path = Path(*zip(*path_coordinates))
    
    earthquake_terrestrial = EarthquakeTerrestrial(parameters)
    displacements_local_terrestrial, displacements_global, displacements_projected, strains_terrestrial = earthquake_terrestrial.request_fibre_strains(path, None, None, True, True, True, parameters.getint('EARTHQUAKE', 'batch_size_sparse'), parameters.getint('EARTHQUAKE', 'worker_count'), parameters.getfloat('EARTHQUAKE', 'request_delay'))
    assert displacements_local_terrestrial.shape[0] == path.vertex_count, f"path vertex count should match the first dimension of displacements_local_terrestrial, but these were {path.vertex_count} and {displacements_local_terrestrial.shape}"
    assert displacements_local_terrestrial.shape[2] == 3, f"displacements_local_terrestrial should have three channels (normal, longitude, latitude), but had {displacements_local_terrestrial.shape[2]}"
    assert displacements_global.shape[0] == path.vertex_count, f"path vertex count should match the first dimension of displacements_global, but these were {path.vertex_count} and {displacements_global.shape[0]}"
    assert displacements_global.shape[2] == 3, f"displacements_global should have three channels (normal, longitude, latitude), but had {displacements_global.shape[2]}"
    assert displacements_local_terrestrial.shape == displacements_global.shape, f"displacements_local_terrestrial and displacements_global should have the same shapes, but had shapes {displacements_local_terrestrial.shape} and {displacements_global.shape}"
    assert displacements_projected.shape == (displacements_local_terrestrial.shape[0] - 1, displacements_local_terrestrial.shape[1], 2), f"displacements_projected and displacements_local_terrestrial should have shapes [S, T, 2] and [S + 1, T, 3], but had shapes {displacements_projected.shape} and {displacements_local_terrestrial.shape}"
    assert strains_terrestrial.shape == (*displacements_projected.shape[:-1], 1), f"strains_terrestrial and displacements_projected should have shapes [S, T, 1] and [S, T, 2], but had shapes {strains_terrestrial.shape} and {displacements_projected.shape}"

    earthquake_submarine = EarthquakeSubmarine(parameters)
    displacements_local_submarine, normal_accelerations, differential_pressures, strains_submarine = earthquake_submarine.request_fibre_strains(path, None, None, True, True, True, parameters.getint('EARTHQUAKE', 'batch_size_sparse'), parameters.getint('EARTHQUAKE', 'worker_count'), parameters.getfloat('EARTHQUAKE', 'request_delay'))
    assert displacements_local_submarine.shape[0] == path.edge_count, f"path edge count should match the first dimension of displacements_local_submarine, but these were {path.edge_count} and {displacements_local_submarine.shape}"
    assert displacements_local_submarine.shape[2] == 3, f"displacements_local_submarine should have three channels (normal, longitude, latitude), but had {displacements_local_submarine.shape[2]}"
    assert normal_accelerations.shape[0] == path.edge_count, f"path edge count should match the first dimension of normal_accelerations, but these were {path.edge_count} and {normal_accelerations.shape[0]}"
    assert normal_accelerations.shape[2] == 1, f"normal_accelerations should have one channel (normal), but had {normal_accelerations.shape[2]}"
    assert displacements_local_submarine.shape[:2] == normal_accelerations.shape[:2], f"displacements_local_terrestrial and normal_accelerations should have shapes [S, T, 3] and [S, T, 1], but had shapes {displacements_local_submarine.shape} and {normal_accelerations.shape}"
    assert differential_pressures.shape == normal_accelerations.shape, f"differential_pressures and normal_accelerations should have the same shapes, but had shapes {differential_pressures.shape} and {normal_accelerations.shape}"
    assert strains_submarine.shape == differential_pressures.shape, f"strains_submarine and differential_pressures should have the same shapes, but had shapes {strains_submarine.shape} and {differential_pressures.shape}"

    assert not np.allclose(strains_submarine.samples_time, strains_terrestrial.samples_time), f"Strains in terrestrial and submarine fibres were the same for the same earthquake, but should be different."

def test_concurrency():
    path = Path(*zip(*path_coordinates))
    earthquake = EarthquakeTerrestrial(parameters)

    fibre_strains_sequential, = earthquake.request_fibre_strains(path, None, None, False, False, False, parameters.getint('EARTHQUAKE', 'batch_size_sparse'), 1, 0)
    fibre_strains_concurrent, = earthquake.request_fibre_strains(path, None, None, False, False, False, 1, parameters.getint('EARTHQUAKE', 'worker_count'), parameters.getfloat('EARTHQUAKE', 'request_delay'))
    
    assert np.allclose(fibre_strains_sequential.samples_time, fibre_strains_concurrent.samples_time), f"Earthquake strains must match when requested sequentially or concurrently, but didn't"

def test_interpolation():
    path = Path(*zip(*path_coordinates)).interpolated(parameters.getfloat('EARTHQUAKE', 'step_length_dense'))

    earthquake_terrestrial = EarthquakeTerrestrial(parameters)
    displacements_global_interpolated, displacements_projected_interpolated, fibre_strains_interpolated = earthquake_terrestrial.request_fibre_strains(
            path,
            parameters.getfloat('EARTHQUAKE', 'step_length_sparse'),
            None,
            False,
            True,
            True,
            parameters.getint('EARTHQUAKE', 'batch_size_sparse'),
            parameters.getint('EARTHQUAKE', 'worker_count'),
            parameters.getfloat('EARTHQUAKE', 'request_delay')
        )

    displacements_global_raw, displacements_projected_raw, fibre_strains_raw = earthquake_terrestrial.request_fibre_strains(
            path,
            None,
            None,
            False,
            True,
            True,
            parameters.getint('EARTHQUAKE', 'batch_size_dense'),
            parameters.getint('EARTHQUAKE', 'worker_count'),
            parameters.getfloat('EARTHQUAKE', 'request_delay')
        )

    earthquake_submarine = EarthquakeSubmarine(parameters)
    normal_accelerations_interpolated, differential_pressures_interpolated, fibre_strains_interpolated = earthquake_submarine.request_fibre_strains(
            path,
            parameters.getfloat('EARTHQUAKE', 'step_length_sparse'),
            None,
            False,
            True,
            True,
            parameters.getint('EARTHQUAKE', 'batch_size_sparse'),
            parameters.getint('EARTHQUAKE', 'worker_count'),
            parameters.getfloat('EARTHQUAKE', 'request_delay')
        )

    normal_accelerations_raw, differential_pressures_raw, fibre_strains_raw = earthquake_submarine.request_fibre_strains(
            path,
            None,
            None,
            False,
            True,
            True,
            parameters.getint('EARTHQUAKE', 'batch_size_dense'),
            parameters.getint('EARTHQUAKE', 'worker_count'),
            parameters.getfloat('EARTHQUAKE', 'request_delay')
        )


    # perturbation_interpolated = earthquake.get_perturbation(
    #         path,
    #         parameters.getfloat('EARTHQUAKE', 'step_length_sparse'),
    #         parameters.getint('EARTHQUAKE', 'batch_length_sparse'),
    #         parameters.getint('EARTHQUAKE', 'worker_count'),
    #         parameters.getfloat('EARTHQUAKE', 'request_delay')
    #     )
    # perturbation_raw = earthquake.get_perturbation(
    #         path,
    #         None,
    #         parameters.getint('EARTHQUAKE', 'batch_size_dense'),
    #         parameters.getint('EARTHQUAKE', 'worker_count'),
    #         parameters.getfloat('EARTHQUAKE', 'request_delay')
    #     )

    # assert perturbation_interpolated.shape == perturbation_raw.shape, f"Synthesised and interpolated earthquake strains must have the same shapes, but these were {perturbation_raw.shape} and {perturbation_interpolated.shape}"
    
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(3, 1)
    # ax[0].grid(True)
    # ax[0].plot(displacements_global_raw.samples_time[:, 10000], 'b-')
    # ax[0].plot(displacements_global_interpolated.samples_time[:, 10000], 'r-x')

    # ax[1].grid(True)
    # ax[1].plot(displacements_projected_raw.samples_time[:, 10000], 'b-')
    # ax[1].plot(displacements_projected_interpolated.samples_time[:, 10000], 'r-x')

    # ax[2].grid(True)
    # ax[2].plot(strains_projected_raw.samples_time[:, 10000], 'b-')
    # ax[2].plot(strains_projected_interpolated.samples_time[:, 10000], 'r-x')
    # fig.show()

    # import pdb
    # pdb.set_trace()

    # assert np.allclose(perturbation_interpolated.strains, perturbation_raw.strains), f"Synthesised and interpolated earthquake strains must match, but didn't"

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