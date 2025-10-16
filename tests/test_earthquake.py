"""
Test correctness of earthquake.py
"""
from configparser import ConfigParser

from tremor_waveplate_toolbox import Earthquake, FibreCoarseStep

parameters = ConfigParser()

parameters['FIBRE'] = {
    'correlation_length': '0.1',   # Correlation length in km
    'beat_length': '0.03',         # Beat length in km
    'section_length': '0.1',       # Section length in km
    'path_coordinates': '[\
        [102.57171090634661, 5.791616724837154],\
        [102.72290646910318, 5.906566563564761],\
        [102.75470997931428, 5.902883822383411]\
    ]', # Coordinates of the Besut-Perhentian Islands cable, taken from https://www.submarinecablemap.com/api/v3/cable/cable-geo.json
    'PMD_parameter': '0.1',      # Polarisation mode dispersion parameter in ps / (km ^ 0.5)
    'realisation_count': '1000', # Number of fibre realisations to simulate simultaneously
    'photoelasticity': '0.1'     # Photoelasticity, which relates material strain to optical strain
}

parameters['EARTHQUAKE'] = {
    'event': 'GCMT:C201002270634A', # A historic earthquake event, structured <catalog>:<identifier> (e.g. from https://www.globalcmt.org/)
    'model': 'ak135f_5s'             # Earth model for Syngine to use from https://ds.iris.edu/ds/products/syngine/#earth
}

def test_earthquake():
    fibre      = FibreCoarseStep(parameters)
    earthquake = Earthquake(parameters)

    times, displacements_normal, displacements_longitude, displacements_latitude, displacements_projected, strains_projected = earthquake.request_fibre_section_projected_strain(fibre, True)
    assert len(times.shape) == 1, f"times should have one dimension, but had shape {times.shape}"
    assert fibre.section_count + 1 == displacements_normal.shape[0], f"fibre section count + 1 should match the first dimension of displacements_normal, but these were {fibre.section_count} and {displacements_normal.shape}"
    assert times.shape[0] == displacements_normal.shape[1], f"times length should match the second dimension of displacements_normal, but these were {len(times)} and {displacements_normal.shape}"
    assert displacements_normal.shape == displacements_longitude.shape and displacements_normal.shape == displacements_latitude.shape, f"displacements_normal, displacements_longitude, and displacements_latitude should have the same shapes, but had shapes {displacements_normal.shape}, {displacements_longitude.shape} and {displacements_latitude.shape}"
    assert displacements_projected.shape == (displacements_normal.shape[0] - 1, displacements_normal.shape[1], 2), f"displacements_projected and displacements_normal should have shapes [S, T, 2] and [S + 1, T], but had shapes {displacements_projected.shape} and {displacements_normal.shape}"
    assert strains_projected.shape == displacements_projected.shape[:-1], f"strains_projected and displacements_projected should have shapes [S, T] and [S, T, 2], but had shapes {strains_projected.shape} and {displacements_projected.shape}"
