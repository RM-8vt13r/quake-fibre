"""
Test correctness of waveplate.py
"""

from configparser import ConfigParser

import numpy as np
import scipy as sp

from tremor_waveplate_toolbox import Waveplate, Transmitter

parameters = ConfigParser()
parameters["TRANSCEIVER"] = {
    "constellation": "QPSK",  # The symbol constellation to use
    "power": "2",             # Transmission power in dBm
    "baud_rate": "1e6",       # Baud rate in symbols / s
    "pulse": "RRCOS",         # Pulseshape, can be SINC or RRCOS, or define your own using the Pulse class
    "pulse_parameter": "0.5", # Parameter to pass to the pulse constructor. For a RRCOS pulse, this is the rolloff factor
    "upsample_factor": "4"    # Samples per symbol
}

# parameters["FIBRE"] = {
#     "Lc": "0.1",   # Correlation length in km
#     "Nsec": "10000", # Number of fibre sections, each of which has length Lc
#     "tau": "0.1",  # Polarisation mode dispersion parameter in ps / (km ^ 0.5)
#     "Nreal": "1000", # Number of fibre realisations to simulate simultaneously
#     "xi": "0.1"    # Photoelasticity, which relates material strain to optical strain
# }

parameters["FIBRE"] = {
    "section_length": "0.1",     # Correlation length in km
    "section_count": "1000",     # Number of fibre sections, each of which has length Lc
    "PMD_parameter": "0.1",      # Polarisation mode dispersion parameter in ps / (km ^ 0.5)
    "realisation_count": "1000", # Number of fibre realisations to simulate simultaneously
    "photoelasticity": "0.1"     # Photoelasticity, which relates material strain to optical strain
}

parameters["SIGNAL"] = {
    "symbol_count": "1e2"
}

def test_waveplate_propagation():
    channel = Waveplate(parameters)

    Tx = Transmitter(parameters)
    _, signal = Tx.transmit_random((1, int(parameters.getfloat("SIGNAL", "symbol_count")),))

    propagated_signal = channel(signal, True)
    assert np.allclose(signal.power_W, propagated_signal.power_W), f"Waveplate model did not retain signal energy"
    assert not np.allclose(signal.samples_time, propagated_signal.samples_time), f"Waveplate model output matched the input (but shouldn't)"
    assert np.allclose(np.einsum('rspq,rbsq->rbsp', channel.PMD_jones(signal.frequency_angular, verbose = True), signal.samples_frequency), propagated_signal.samples_frequency), f"Waveplate model propagation and Jones matrix produced different results"

    channel.section_material_strain = np.random.default_rng().normal(0, 10, channel.section_count)

    propagated_signal_earthquake = channel(signal, True)
    assert np.allclose(propagated_signal_earthquake.power_W, signal.power_W), f"Earthquake-perturbed waveplate model did not retain signal energy"

    assert np.allclose(signal.power_W, propagated_signal_earthquake.power_W), f"Earthquake-perturbed waveplate model did not retain signal energy"
    assert not np.allclose(signal.samples_time, propagated_signal.samples_time), f"Earthquake-perturbed waveplate model output matched the input (but shouldn't)"
    assert not np.allclose(propagated_signal.samples_time, propagated_signal_earthquake.samples_time), f"Propagated signal before and after the earthquake were the same, but should be different"
    assert np.allclose(np.einsum('rspq,rbsq->rbsp', channel.PMD_jones(signal.frequency_angular, verbose = True), signal.samples_frequency), propagated_signal_earthquake.samples_frequency), f"Earthquake-perturbed waveplate model propagation and Jones matrix produced different results"


def test_waveplate_SOP_rotation_setter():
    channel = Waveplate(parameters)

    channel_section_SOP_rotation_stokes = channel.section_SOP_rotation_stokes.copy()
    channel_section_SOP_rotation = channel.section_SOP_rotation.copy()

    channel.section_SOP_rotation_stokes = channel_section_SOP_rotation_stokes
    assert np.allclose(channel.section_SOP_rotation, channel_section_SOP_rotation), f"section_SOP_rotation not updated correctly after setting section_SOP_rotation_stokes"


def test_waveplate_initialisation():
    channel = Waveplate(parameters)

    assert np.allclose(np.conj(channel.section_SOP_rotation).swapaxes(-2, -1) @ channel.section_SOP_rotation, np.eye(2)), f"Waveplate model PSP rotation matrices are not unitary"

    DGD_accumulated = channel.DGD
    assert np.isclose(np.mean(DGD_accumulated), channel.PMD_parameter * np.sqrt(channel.length), rtol = 1e-1), f"Accumulated DGD does not match PMD parameter"

    # # Check if rotation is uniformly distributed
    # accumulated_rotation_distribution, rotation_bin_edges = np.histogram(
    #     accumulated_rotation,
    #     bins = 10,
    #     range = (0, np.pi),
    #     density = True
    # )
    # rotation_bin_values = (rotation_bin_edges[:-1] + rotation_bin_edges[1:]) / 2

    # Check if accumulated DGD is Maxwellian-distributed.
    # Curti et al. - Statistical Treatment of the Evolution of the Principal States of Polarization in Single-Mode Fibers
    DGD_accumulated_histogram, DGD_accumulated_bin_edges = np.histogram(
        DGD_accumulated,
        bins = 100,
        range = (0, np.mean(DGD_accumulated) * 5),
        density = False
    )
    DGD_accumulated_histogram = DGD_accumulated_histogram / (len(DGD_accumulated) * np.diff(DGD_accumulated_bin_edges))
    DGD_accumulated_bin_values = (DGD_accumulated_bin_edges[:-1] + DGD_accumulated_bin_edges[1:]) / 2

    q = channel.PMD_parameter * np.sqrt(channel.length * np.pi / 8)
    DGD_accumulated_distribution = 2 * DGD_accumulated_bin_values ** 2 / (np.sqrt(2 * np.pi) * q ** 3) * np.exp(-DGD_accumulated_bin_values ** 2 / (2 * q ** 2))

    assert np.abs(np.sum(np.where(np.isclose(DGD_accumulated_histogram, 0), DGD_accumulated_distribution, sp.special.rel_entr(DGD_accumulated_distribution, DGD_accumulated_histogram)))) < 1, f"Accumulated DGD does not approach correct Maxwell-Bolzmann distribution"

    # import matplotlib.pyplot as plt
    # # fig1, ax1 = plt.subplots()
    # # ax1.hist(accumulated_rotation_distribution, bins = rotation_bin_edges)
    # # ax1.plot((0, np.pi), (1 / np.pi, 1 / np.pi))
    # # ax1.set_title("Accumulated rotation per realisation")
    # # ax1.set_xlabel("Rotation [Rad]")
    # # ax1.set_ylabel("Probability")

    # fig2, ax2 = plt.subplots()
    # ax2.bar(
    #     x = DGD_accumulated_bin_values,
    #     height = DGD_accumulated_histogram,
    #     width = np.diff(DGD_accumulated_bin_edges)
    # )
    # ax2.plot(DGD_accumulated_bin_values, DGD_accumulated_distribution, 'r')
    # ax2.set_title("Accumulated DGD per realisation")
    # ax2.set_xlabel("DGD [ps]")
    # ax2.set_ylabel("Probability")

    # plt.show()

    # import pdb
    # pdb.set_trace()
