"""
Test correctness of fibre.py
"""

from configparser import ConfigParser
import copy
import sys

try:
    import cupy as cp
except:
    print("cupy not available; skipping all CUDA tests..")
import numpy as np
import scipy as sp

from tremor_waveplate_toolbox import FibreCoarseStep, FibreMarcuse, Transmitter, Device

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
    'beat_length': '0.05',       # Beat length in km
    'section_length': '0.00167', # section length in km
    # 'section_length': '0.1', # section length in km
    'section_count': '1000',     # Number of fibre sections, each of which has length Lc
    'PMD_parameter': '10',       # Polarisation mode dispersion parameter in ps / (km ^ 0.5)
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
    [102.58683046262226, 5.803111708709915]\
]' # Coordinates along the Besut-Perhentian Islands cable, taken from https://www.submarinecablemap.com/api/v3/cable/cable-geo.json

def test_fibre_propagation():
    for channel in (FibreMarcuse(parameters), FibreCoarseStep(parameters)):
        try:
            channel.path_coordinates
            channel.path_lengths
            channel.path_positions
            channel.section_coordinates
        except: pass
        else:   raise AssertionError(f"{type(channel)} should raise an error when accessing unset path_coordinates, path_lengths, path_positions or section_coordinates, but didn't")

        assert np.all(channel.section_lengths == parameters.getfloat('FIBRE', 'section_length')), f"{type(channel)} sections should have length section_length, but didn't"
        assert channel.section_count == parameters.getint('FIBRE', 'section_count'), f"{type(channel)} should have {parameters.getint('FIBRE', 'section_count')} sections, but had {channel.section_count}"

        transmitter = Transmitter(parameters)
        _, signal = transmitter.transmit_random(1, int(parameters.getfloat("SIGNAL", "symbol_count")))

        # Test signal propagation and Jones matrix construction
        propagated_signal = channel(signal, verbose = True)
        assert np.allclose(signal.power_W, propagated_signal.power_W), f"{type(channel)} did not retain signal energy"
        assert not np.allclose(signal.samples_time, propagated_signal.samples_time), f"{type(channel)} output matched the input (but shouldn't)"
        jones_matrices = channel.Jones(signal.frequency_angular, verbose = True)
        assert np.allclose(np.einsum('rspq,rbsq->rbsp', jones_matrices, signal.samples_frequency), propagated_signal.samples_frequency), f"{type(channel)} propagation and Jones matrix produced different results"

        # Test the DGD-less case
        channel._PMD_parameter = 0.0
        # import pdb
        # pdb.set_trace()
        propagated_signal_no_DGD = channel(signal, verbose = True)
        assert np.allclose(signal.power_W, propagated_signal_no_DGD.power_W), f"{type(channel)} did not retain signal energy in the DGD-less case"
        assert not np.allclose(signal.samples_time, propagated_signal_no_DGD.samples_time), f"{type(channel)} output matched the input (but shouldn't) in the DGD-less case"
        assert not np.allclose(propagated_signal.samples_time, propagated_signal_no_DGD.samples_time), f"{type(channel)} outputs matched for the cases with and without DGD (but shouldn't)"
        jones_matrices_no_DGD = channel.Jones(signal.frequency_angular, verbose = True)
        assert np.allclose(np.einsum('rspq,rbsq->rbsp', jones_matrices_no_DGD, signal.samples_frequency), propagated_signal_no_DGD.samples_frequency), f"{type(channel)} propagation and Jones matrix produced different results"
        assert not np.allclose(jones_matrices_no_DGD, jones_matrices), f"{type(channel)} Jones matrices matched in the cases with and without DGD (but shouldn't)"
        channel._PMD_parameter = parameters.getfloat('FIBRE', 'PMD_parameter')

        if 'cupy' not in sys.modules: continue

        signal.to_device(Device.CUDA)

        propagated_signal_cuda = channel(signal, verbose = True)
        jones_matrices_cuda = channel.Jones(signal.frequency_angular, verbose = True)

        propagated_signal_cuda.to_device(Device.CPU)
        jones_matrices_cuda = jones_matrices_cuda.get()

        assert np.allclose(propagated_signal_cuda.samples_time, propagated_signal.samples_time), f"{type(channel)} CUDA- and CPU propagation yielded deviating results"
        assert np.allclose(jones_matrices_cuda, jones_matrices), f"{type(channel)} CUDA- and CPU Jones matrices yielded deviating results"

        channel._PMD_parameter = 0.0
        propagated_signal_no_DGD_cuda = channel(signal, verbose = True)
        jones_matrices_no_DGD_cuda = channel.Jones(signal.frequency_angular, verbose = True)
        channel._PMD_parameter = parameters.getfloat('FIBRE', 'PMD_parameter')

        propagated_signal_no_DGD_cuda.to_device(Device.CPU)
        jones_matrices_no_DGD_cuda = jones_matrices_no_DGD_cuda.get()

        assert np.allclose(propagated_signal_no_DGD_cuda.samples_time, propagated_signal_no_DGD.samples_time), f"{type(channel)} CUDA- and CPU propagation yielded deviating results in the DGD-less case"
        assert np.allclose(jones_matrices_no_DGD_cuda, jones_matrices_no_DGD), f"{type(channel)} CUDA- and CPU Jones matrices yielded deviating results in the DGD-less case"
        
        signal.to_device(Device.CPU)

    if isinstance(channel, FibreCoarseStep): return

    if 'cupy' in sys.modules: signal.to_device(Device.CUDA)

    material_strain = Signal(
        samples = 10 * np.random.default_rng().normal(size = (channel.section_count, 60, 1)),
        sample_rate = 1,
    )
    propagated_signal_earthquake_0s = channel(signal, strain = material_strain, transmission_start_time = 0, verbose = True)
    propagated_signal_earthquake_30s = channel(signal, strain = material_strain, transmission_start_time = 30, verbose = True)
    jones_matrices_earthquake_0s = channel.Jones(signal.frequency_angular, strain = material_strain, transmission_start_time = 0, verbose = True)
    jones_matrices_earthquake_30s = channel.Jones(signal.frequency_angular, strain = material_strain, transmission_start_time = 30, verbose = True)

    propagated_signal_earthquake_0s.to_device(Device.CPU)
    propagated_signal_earthquake_30s.to_device(Device.CPU)
    jones_matrices_earthquake_0s = jones_matrices_earthquake_0s.get()
    jones_matrices_earthquake_30s = jones_matrices_earthquake_30s.get()

    assert np.allclose(propagated_signal_earthquake_0s.power_W, signal.power_W), f"{type(channel)} earthquake-perturbed fibre at 0 seconds did not retain signal energy"
    assert np.allclose(propagated_signal_earthquake_30s.power_W, signal.power_W), f"{type(channel)} earthquake-perturbed fibre at 30 seconds did not retain signal energy"
    assert not np.allclose(propagated_signal_earthquake_0s.samples_time, signal.samples_time), f"{type(channel)} earthquake-perturbed fibre at 0 seconds matches input signal (but shouldn't)"
    assert not np.allclose(propagated_signal_earthquake_30s.samples_time, signal.samples_time), f"{type(channel)} earthquake-perturbed fibre at 30 seconds matches input signal (but shouldn't)"
    assert not np.allclose(propagated_signal_earthquake_0s.samples_time, propagated_signal.samples_time), f"{type(channel)} earthquake didn't perturb signal at 0 seconds"
    assert not np.allclose(propagated_signal_earthquake_30s.samples_time, propagated_signal.samples_time), f"{type(channel)} earthquake didn't perturb signal at 30 seconds"
    assert not np.allclose(propagated_signal_earthquake_0s.samples_time, propagated_signal_earthquake_30s.samples_time), f"{type(channel)} earthquake perturbed signals at 0 and 30 seconds the same (but shouldn't have)"

    assert np.allclose(np.einsum('rspq,rbsq->rbsp', jones_matrices_earthquake_0s, signal.samples_frequency), propagated_signal_earthquake_0s.samples_frequency), f"{type(channel)} earthquake-perturbed fibre propagation and Jones matrix at 0 seconds produced different results"
    assert np.allclose(np.einsum('rspq,rbsq->rbsp', jones_matrices_earthquake_30s, signal.samples_frequency), propagated_signal_earthquake_30s.samples_frequency), f"{type(channel)} earthquake-perturbed fibre propagation and Jones matrix at 30 seconds produced different results"


# def test_fibre_PSP_setter():
#     channel = Fibre(parameters)

#     channel_section_SOP_rotation_stokes = channel.section_SOP_rotation_stokes.copy()
#     channel_section_SOP_rotation = channel.section_SOP_rotation.copy()

#     channel.section_SOP_rotation_stokes = channel_section_SOP_rotation_stokes
#     assert np.allclose(channel.section_SOP_rotation, channel_section_SOP_rotation), f"section_SOP_rotation not updated correctly after setting section_SOP_rotation_stokes"


def test_fibre_initialisation():
    for channel in (FibreMarcuse(parameters), FibreCoarseStep(parameters)):
        DGD_accumulated = channel.DGD
        assert np.isclose(np.mean(DGD_accumulated), channel.PMD_parameter * np.sqrt(channel.length), rtol = 1e-1), f"Accumulated DGD does not match PMD parameter"

        if isinstance(channel, FibreMarcuse): continue
        # Check if accumulated DGD is Maxwellian-distributed.
        # Curti et al. - Statistical Treatment of the Evolution of the Principal States of Polarization in Single-Mode Fibers
        assert sp.stats.kstest(DGD_accumulated, lambda x: sp.stats.maxwell.cdf(x, scale = channel.PMD_parameter * np.sqrt(channel.length * np.pi / 8))).pvalue > 0.05, "Accumulated DGD does not approach correct Maxwell-Bolzmann distribution"
        assert np.allclose(channel.section_PSP.swapaxes(-2, -1).conjugate() @ channel.section_PSP, np.eye(2)), f"Fibre PSP rotation matrices are not unitary"
        

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

        if 'cupy' in sys.modules: signal.to_device(Device.CUDA)

        propagated_signal = channel(signal, verbose = True)
        assert signal.xp.allclose(signal.power_W, propagated_signal.power_W), f"Fibre did not retain signal energy"
        assert not signal.xp.allclose(signal.samples_time, propagated_signal.samples_time), f"Fibre output matched the input (but shouldn't)"
        jones_matrices = channel.Jones(signal.frequency_angular, verbose = True)
        assert signal.xp.allclose(signal.xp.einsum('rspq,rbsq->rbsp', jones_matrices, signal.samples_frequency), propagated_signal.samples_frequency), f"Fibre propagation and Jones matrix produced different results"


def test_fibre_dict():
    for channel in (FibreMarcuse(parameters), FibreCoarseStep(parameters)):
        fibre_dict = channel.to_dict()
        channel_copy = type(channel).from_dict(fibre_dict)
        assert channel == channel_copy, f"Channel was not copied properly using to_dict and from_dict"

    for channel in (FibreMarcuse(parameters_geographic), FibreCoarseStep(parameters_geographic)):
        fibre_dict = channel.to_dict()
        channel_copy = type(channel).from_dict(fibre_dict)
        assert channel == channel_copy, f"Geographic channel was not copied propertly using to_dict and from_dict"