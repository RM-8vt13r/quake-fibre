"""
Test correctness of fibre.py
"""

from configparser import ConfigParser
import copy

import numpy as np
import scipy as sp

from tremor_waveplate_toolbox import FibreCoarseStep, FibreMarcuse, Transmitter

parameters = ConfigParser()
parameters['TRANSCEIVER'] = {
    'constellation': 'QPSK',  # The symbol constellation to use
    'power': '2',             # Transmission power in dBm
    'baud_rate': '1e6',       # Baud rate in symbols / s
    'pulse': 'RRCOS',         # Pulseshape, can be SINC or RRCOS, or define your own using the Pulse class
    'pulse_parameter': '0.5', # Parameter to pass to the pulse constructor. For a RRCOS pulse, this is the rolloff factor
    'upsample_factor': '4'    # Samples per symbol
}

parameters['FIBRE'] = {
    'correlation_length': '0.1', # Correlation length in km
    'beat_length': '.05',        # Beat length in km
    'section_length': '0.025',   # section length in km
    'section_count': '1000',     # Number of fibre sections, each of which has length Lc
    'PMD_parameter': '0.1',      # Polarisation mode dispersion parameter in ps / (km ^ 0.5)
    'realisation_count': '999',  # Number of fibre realisations to simulate simultaneously
    'photoelasticity': '0.1'     # Photoelasticity, which relates material strain to optical strain
}

parameters['SIGNAL'] = {
    'symbol_count': '1e2'
}

parameters_geographic = copy.deepcopy(parameters)
parameters_geographic['FIBRE']['section_count'] = 'None'
# parameters_geographic['FIBRE']['path_coordinates'] = '[\
#     [102.57171090634661, 5.791616724837154],\
#     [102.72290646910318, 5.906566563564761],\
#     [102.75470997931428, 5.902883822383411]\
# ]' # Coordinates of the Besut-Perhentian Islands cable, taken from https://www.submarinecablemap.com/api/v3/cable/cable-geo.json

parameters_geographic['FIBRE']['path_coordinates'] = '[\
    [102.57171090634661, 5.791616724837154],\
    [102.72290646910318, 5.906566563564761]\
]' # Coordinates of the Besut-Perhentian Islands cable, taken from https://www.submarinecablemap.com/api/v3/cable/cable-geo.json


def test_fibre_propagation():
    for channel in (FibreMarcuse(parameters), FibreCoarseStep(parameters)):
        try:    channel.path_coordinates
        except: pass
        else:   raise AssertionError("Channel should raise an error when accessing unset path_coordinates, but didn't")

        try:    channel.path_lengths
        except: pass
        else:   raise AssertionError("Channel should raise an error when accessing unset path_lengths, but didn't")

        try:    channel.path_positions
        except: pass
        else:   raise AssertionError("Channel should raise an error when accessing unset path_positions, but didn't")

        try:    channel.section_coordinates
        except: pass
        else:   raise AssertionError("Channel should raise an error when accessing unset section_coordinates, but didn't")

        assert np.all(channel.section_lengths == parameters.getfloat('FIBRE', 'section_length')), "All channel sections should have length section_length, but didn't"
        assert channel.section_count == parameters.getint('FIBRE', 'section_count'), f"Channel should have {parameters.getint('FIBRE', 'section_count')} sections, but had {channel.section_count}"

        transmitter = Transmitter(parameters)
        _, signal = transmitter.transmit_random(1, int(parameters.getfloat("SIGNAL", "symbol_count")))

        propagated_signal = channel(signal, verbose = True)
        assert np.allclose(signal.power_W, propagated_signal.power_W), f"Fibre did not retain signal energy"
        assert not np.allclose(signal.samples_time, propagated_signal.samples_time), f"Fibre output matched the input (but shouldn't)"
        jones_matrices = channel.Jones(signal.frequency_angular, verbose = True)
        assert np.allclose(np.einsum('rspq,rbsq->rbsp', jones_matrices, signal.samples_frequency), propagated_signal.samples_frequency), f"Fibre propagation and Jones matrix produced different results"

    # channel.section_material_strain = np.random.default_rng().normal(0, 10, channel.section_count)

    # propagated_signal_earthquake = channel(signal, True)
    # assert np.allclose(propagated_signal_earthquake.power_W, signal.power_W), f"Earthquake-perturbed fibre did not retain signal energy"

    # assert np.allclose(signal.power_W, propagated_signal_earthquake.power_W), f"Earthquake-perturbed fibre did not retain signal energy"
    # assert not np.allclose(signal.samples_time, propagated_signal.samples_time), f"Earthquake-perturbed fibre output matched the input (but shouldn't)"
    # assert not np.allclose(propagated_signal.samples_time, propagated_signal_earthquake.samples_time), f"Propagated signal before and after the earthquake were the same, but should be different"
    # assert np.allclose(np.einsum('rspq,rbsq->rbsp', channel.PMD_jones(signal.frequency_angular, verbose = True), signal.samples_frequency), propagated_signal_earthquake.samples_frequency), f"Earthquake-perturbed fibre propagation and Jones matrix produced different results"

    # assert np.allclose(np.einsum('rspq,rbsq->rbsp', channel.PMD_jones(signal.frequency_angular, verbose = True), signal.samples_frequency), propagated_signal.samples_frequency), f"Fibre propagation and Jones matrix produced different results"

# def test_fibre_PSP_setter():
#     channel = Fibre(parameters)

#     channel_section_SOP_rotation_stokes = channel.section_SOP_rotation_stokes.copy()
#     channel_section_SOP_rotation = channel.section_SOP_rotation.copy()

#     channel.section_SOP_rotation_stokes = channel_section_SOP_rotation_stokes
#     assert np.allclose(channel.section_SOP_rotation, channel_section_SOP_rotation), f"section_SOP_rotation not updated correctly after setting section_SOP_rotation_stokes"


def test_fibre_initialisation():
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
        
    for channel in (FibreMarcuse(parameters), FibreCoarseStep(parameters)):
        DGD_accumulated = channel.DGD
        assert np.isclose(np.mean(DGD_accumulated), channel.PMD_parameter * np.sqrt(channel.length), rtol = 1e-1), f"Accumulated DGD does not match PMD parameter"

        if isinstance(channel, FibreMarcuse): continue
        # Check if accumulated DGD is Maxwellian-distributed.
        # Curti et al. - Statistical Treatment of the Evolution of the Principal States of Polarization in Single-Mode Fibers
        # DGD_accumulated_histogram, DGD_accumulated_bin_edges = np.histogram(
        #     DGD_accumulated,
        #     bins = 150,
        #     range = (0, np.mean(DGD_accumulated) * 5),
        #     density = False
        # )
        # DGD_accumulated_histogram = DGD_accumulated_histogram / (len(DGD_accumulated) * np.diff(DGD_accumulated_bin_edges))
        # DGD_accumulated_bin_values = (DGD_accumulated_bin_edges[:-1] + DGD_accumulated_bin_edges[1:]) / 2

        # assert np.abs(np.sum(np.where(np.isclose(DGD_accumulated_histogram, 0), DGD_accumulated_distribution, sp.special.rel_entr(DGD_accumulated_distribution, DGD_accumulated_histogram)))) < 1, f"Accumulated DGD does not approach correct Maxwell-Bolzmann distribution"
        # ax.plot(DGD_accumulated_bin_values, DGD_accumulated_histogram)
        # print(sp.stats.kstest(DGD_accumulated, lambda x: sp.stats.maxwell.cdf(x, scale = channel.PMD_parameter * np.sqrt(channel.length * np.pi / 8))))
        assert sp.stats.kstest(DGD_accumulated, lambda x: sp.stats.maxwell.cdf(x, scale = channel.PMD_parameter * np.sqrt(channel.length * np.pi / 8))).pvalue > 0.05, "Accumulated DGD does not approach correct Maxwell-Bolzmann distribution"
        assert np.allclose(channel.section_PSP.swapaxes(-2, -1).conjugate() @ channel.section_PSP, np.eye(2)), f"Fibre PSP rotation matrices are not unitary"
        
    # ax.plot(DGD_accumulated_bin_values, sp.stats.maxwell.pdf(DGD_accumulated_bin_values, scale = channel.PMD_parameter * np.sqrt(channel.length * np.pi / 8)))
    # import pdb
    # pdb.set_trace()
    # fig.show()
        

def test_fibre_path():
    for channel in (FibreMarcuse(parameters_geographic), FibreCoarseStep(parameters_geographic)):
        assert np.allclose(channel.section_lengths[:-1], parameters_geographic.getfloat('FIBRE', 'section_length')), "Fibre section lengths didn't match parameter section_length with geographic initialisation"
        assert np.isclose(np.sum(channel.section_lengths), np.sum(channel.path_lengths)), f"Fibre path- and section lengths add up to {np.sum(channel.path_lengths)} and {np.sum(channel.section_lengths)}, but should have the same total"

        try:
            channel.path_coordinates
            channel.path_lengths
            channel.path_positions
            channel.section_coordinates
        except AttributeError:
            raise AssertionError("Channel should contain path_coordinates, path_lengths, path_positions, and section_coordinates, but didn't")

        transmitter = Transmitter(parameters)
        _, signal = transmitter.transmit_random(1, int(parameters.getfloat("SIGNAL", "symbol_count")))

        propagated_signal = channel(signal, verbose = True)
        assert np.allclose(signal.power_W, propagated_signal.power_W), f"Fibre did not retain signal energy"
        assert not np.allclose(signal.samples_time, propagated_signal.samples_time), f"Fibre output matched the input (but shouldn't)"
        jones_matrices = channel.Jones(signal.frequency_angular, verbose = True)
        assert np.allclose(np.einsum('rspq,rbsq->rbsp', jones_matrices, signal.samples_frequency), propagated_signal.samples_frequency), f"Fibre propagation and Jones matrix produced different results"

        # channel.section_material_strain = np.random.default_rng().normal(0, 10, channel.section_count)

        # propagated_signal_earthquake = channel(signal, True)
        # assert np.allclose(propagated_signal_earthquake.power_W, signal.power_W), f"Earthquake-perturbed fibre did not retain signal energy"

        # assert np.allclose(signal.power_W, propagated_signal_earthquake.power_W), f"Earthquake-perturbed fibre did not retain signal energy"
        # assert not np.allclose(signal.samples_time, propagated_signal.samples_time), f"Earthquake-perturbed fibre output matched the input (but shouldn't)"
        # assert not np.allclose(propagated_signal.samples_time, propagated_signal_earthquake.samples_time), f"Propagated signal before and after the earthquake were the same, but should be different"
        # assert np.allclose(np.einsum('rspq,rbsq->rbsp', channel.PMD_jones(signal.frequency_angular, verbose = True), signal.samples_frequency), propagated_signal_earthquake.samples_frequency), f"Earthquake-perturbed fibre propagation and Jones matrix produced different results"


def test_fibre_dict():
    for channel in (FibreMarcuse(parameters), FibreCoarseStep(parameters)):
        fibre_dict = channel.to_dict()
        channel_copy = type(channel).from_dict(fibre_dict)
        assert channel == channel_copy, f"Channel was not copied properly using to_dict and from_dict"

    for channel in (FibreMarcuse(parameters_geographic), FibreCoarseStep(parameters_geographic)):
        fibre_dict = channel.to_dict()
        channel_copy = type(channel).from_dict(fibre_dict)
        assert channel == channel_copy, f"Geographic channel was not copied propertly using to_dict and from_dict"