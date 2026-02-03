"""
Test correctness of earthquake.py
"""
from configparser import ConfigParser
import json

import numpy as np

from tremor_waveplate_toolbox import Earthquake, Path

parameters = ConfigParser()

path_coordinates = [
    [102.57171090634661, 5.791616724837154],
    [102.72290646910318, 5.906566563564761],
    [102.75470997931428, 5.902883822383411]
]

parameters['EARTHQUAKE'] = {
    'event': 'GCMT:C201002270634A', # A historic earthquake event, structured <catalog>:<identifier> (e.g. from https://www.globalcmt.org/)
    'model': 'ak135f_5s',           # Earth model for Syngine to use from https://ds.iris.edu/ds/products/syngine/#earth
    'batch_size_sparse': '6',       # The number of seismograms to request at most simultaneously
    'batch_size_dense': '50',       # The number of seismograms to request at most simultaneously
    'step_length_sparse': '1',      # Distances at which to request seismograms in km; these seismograms are interpolated to all fibre section 
    'step_length_dense': '0.01',    # Distances at which to request seismograms in km; these seismograms are interpolated to all fibre section 
    'worker_count': '5',            # Number of threads to request seismograms from Syngine simultaneously, high numbers (>20) might yield a temporary block
    'request_delay': '0.1'          # Minimum delay between filing concurrent seismogram requests (guidelines at https://ds.iris.edu/ds/nodes/dmc/services/usage/)
}

def test_earthquake():
    path = Path(*zip(*path_coordinates))
    earthquake = Earthquake(parameters)

    displacements_local, displacements_global, displacements_projected, strains_projected = earthquake.request_projected_strains(path, None, None, True, True, True, parameters.getint('EARTHQUAKE', 'batch_size_sparse'), parameters.getint('EARTHQUAKE', 'worker_count'), parameters.getfloat('EARTHQUAKE', 'request_delay'))
    # assert len(times.shape) == 1, f"times should have one dimension, but had shape {times.shape}"
    assert displacements_local.shape[0] == path.vertex_count, f"path vertex count should match the first dimension of displacements_local, but these were {path.vertex_count} and {displacements_local.shape}"
    assert displacements_local.shape[2] == 3, f"displacements_local should have three channels (normal, longitude, latitude), but had {displacements_local.shape[2]}"
    assert displacements_global.shape[0] == path.vertex_count, f"path vertex count should match the first dimension of displacements_global, but these were {path.vertex_count} and {displacements_global.shape}"
    assert displacements_global.shape[2] == 3, f"displacements_global should have three channels (normal, longitude, latitude), but had {displacements_global.shape[2]}"
    # assert times.shape[0] == displacements_normal.shape[1], f"times length should match the second dimension of displacements_normal, but these were {len(times)} and {displacements_normal.shape}"
    # assert displacements_normal.shape == displacements_longitude.shape and displacements_normal.shape == displacements_latitude.shape, f"displacements_normal, displacements_longitude, and displacements_latitude should have the same shapes, but had shapes {displacements_normal.shape}, {displacements_longitude.shape} and {displacements_latitude.shape}"
    assert displacements_local.shape == displacements_global.shape, f"displacements_local and displacements_global should have the same shapes, but had shapes {displacements_local.shape} and {displacements_global.shape}"
    assert displacements_projected.shape == (displacements_local.shape[0] - 1, displacements_local.shape[1], 2), f"displacements_projected and displacements_local should have shapes [S, T, 2] and [S + 1, T, 3], but had shapes {displacements_projected.shape} and {displacements_local.shape}"
    assert strains_projected.shape == (*displacements_projected.shape[:-1], 1), f"strains_projected and displacements_projected should have shapes [S, T, 1] and [S, T, 2], but had shapes {strains_projected.shape} and {displacements_projected.shape}"

def test_concurrency():
    path = Path(*zip(*path_coordinates))
    earthquake = Earthquake(parameters)

    strains_projected_sequential, = earthquake.request_projected_strains(path, None, None, False, False, False, parameters.getint('EARTHQUAKE', 'batch_size_sparse'), 1, 0)
    strains_projected_concurrent, = earthquake.request_projected_strains(path, None, None, False, False, False, 1, parameters.getint('EARTHQUAKE', 'worker_count'), parameters.getfloat('EARTHQUAKE', 'request_delay'))
    
    assert np.allclose(strains_projected_sequential.samples_time, strains_projected_concurrent.samples_time), f"Earthquake strains must match when requested sequentially or concurrently, but didn't"

def test_interpolation():
    path = Path(*zip(*path_coordinates)).interpolated(parameters.getfloat('EARTHQUAKE', 'step_length_dense'))
    earthquake = Earthquake(parameters)

    displacements_global_interpolated, displacements_projected_interpolated, strains_projected_interpolated = earthquake.request_projected_strains(
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

    displacements_global_raw, displacements_projected_raw, strains_projected_raw = earthquake.request_projected_strains(
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
    earthquake = Earthquake(parameters)

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