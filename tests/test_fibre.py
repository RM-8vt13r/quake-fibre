"""
Test correctness of fibre.py
"""

from configparser import ConfigParser
import copy
import sys
import logging
logging.basicConfig(level = logging.INFO)

try:
    import cupy as cp
except:
    print("cupy not available; skipping all CUDA tests..")
import numpy as np
import scipy as sp

from tremor_waveplate_toolbox import FibreCoarseStep, FibreMarcuse, Transmitter, Device, Signal, Perturbation

parameters = ConfigParser()
parameters['TRANSCEIVER'] = {
    'constellation': 'QPSK',  # The symbol constellation to use
    'power': '2',             # Transmission power in dBm
    'baud_rate': '40e9',      # Baud rate in symbols / s
    'pulse': 'RRCOS',         # Pulseshape, can be SINC or RRCOS, or define your own using the Pulse class
    'pulse_parameter': '0.5', # Parameter to pass to the pulse constructor. For a RRCOS pulse, this is the rolloff factor
    'upsample_factor': '4'    # Samples per symbol
}

parameters['FIBRE'] = {
    'correlation_length': '0.1',          # Correlation length in km
    'beat_length': '0.05',                # Beat length in km
    'section_length': '0.0167',           # section length in km
    'section_count': '1000',              # Number of fibre sections
    'modulus_model': 'FIXED',             # Polarisation mode dispersion initialisation model
    'chromatic_dispersion': '0',          # Chromatic dispersion parameter in ps ^ 2 / km
    'nonlinearity': '0',                  # Nonlinearity parameter in 1 / (W km)
    'polarisation_mode_dispersion': '10', # Polarisation mode dispersion parameter in ps / (km ^ 0.5); If 0, turns off major axes rotations as well
    'realisation_count': '99',            # Number of fibre realisations to simulate simultaneously
    'photoelasticity': '0.78'             # Photoelasticity, which relates material strain to optical strain
}

parameters['SIGNAL'] = {
    'symbol_count': '1e2' # How many symbols to transmit
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
    for channel in (FibreCoarseStep(parameters), FibreMarcuse(parameters)):
        try:
            channel.path.coordinates
        except: pass
        else:   raise AssertionError(f"{type(channel)} should raise an error when accessing unset path.coordinates, but didn't")
        try:
            channel.section_path.coordinates
        except: pass
        else:   raise AssertionError(f"{type(channel)} should raise an error when accessing unset section_path.coordinates, but didn't")

        assert np.all(channel.section_path.lengths == parameters.getfloat('FIBRE', 'section_length')), f"{type(channel)} sections should have length section_length, but didn't"
        assert channel.section_path.edge_count == parameters.getint('FIBRE', 'section_count'), f"{type(channel)} should have {parameters.getint('FIBRE', 'section_count')} sections, but had {channel.section_path.edge_count}"

        transmitter = Transmitter(parameters)
        _, signal = transmitter.transmit_random(1, int(parameters.getfloat("SIGNAL", "symbol_count")))

        # Test signal propagation and Jones matrix construction
        propagated_signal = channel(signal)
        assert np.allclose(signal.power_W, propagated_signal.power_W), f"{type(channel)} did not retain signal energy"
        assert not np.allclose(signal.samples_time, propagated_signal.samples_time), f"{type(channel)} output matched the input (but shouldn't)"
        jones_matrices = channel.Jones(signal.frequency_angular)
        assert np.allclose(np.einsum('rbspq,rbsq->rbsp', jones_matrices, signal.samples_frequency), propagated_signal.samples_frequency), f"{type(channel)} propagation and Jones matrix produced different results"

        # Test the case without differential group delay
        channel._polarisation_mode_dispersion = 0.0
        propagated_signal_no_DGD = channel(signal)
        assert np.allclose(signal.power_W, propagated_signal_no_DGD.power_W), f"{type(channel)} did not retain signal energy in the case without polarisation mode dispersion"
        assert np.allclose(signal.samples_time, propagated_signal_no_DGD.samples_time), f"{type(channel)} output didn't match the input (but should) in the case without polarisation mode dispersion"
        assert not np.allclose(propagated_signal.samples_time, propagated_signal_no_DGD.samples_time), f"{type(channel)} outputs matched for the cases with and without polarisation mode dispersion (but shouldn't)"
        jones_matrices_no_DGD = channel.Jones(signal.frequency_angular)
        assert np.allclose(np.einsum('rbspq,rbsq->rbsp', jones_matrices_no_DGD, signal.samples_frequency), propagated_signal_no_DGD.samples_frequency), f"{type(channel)} propagation and Jones matrix produced different results"
        assert not np.allclose(jones_matrices_no_DGD, jones_matrices), f"{type(channel)} Jones matrices matched in the cases with and without differential group delay (but shouldn't)"
        channel._polarisation_mode_dispersion = parameters.getfloat('FIBRE', 'polarisation_mode_dispersion')

        # Test chromatic dispersion, nonlinearity, polarisation-dependent loss and amplified spontaneous emission..

        # Tests CUDA implementation
        if 'cupy' not in sys.modules: continue

        signal.to_device(Device.CUDA)

        propagated_signal_cuda = channel(signal)
        jones_matrices_cuda = channel.Jones(signal.frequency_angular)

        propagated_signal_cuda.to_device(Device.CPU)
        jones_matrices_cuda = jones_matrices_cuda.get()

        assert np.allclose(propagated_signal_cuda.samples_time, propagated_signal.samples_time), f"{type(channel)} CUDA- and CPU propagation yielded deviating results"
        assert np.allclose(jones_matrices_cuda, jones_matrices), f"{type(channel)} CUDA- and CPU Jones matrices yielded deviating results"

        channel._polarisation_mode_dispersion = 0.0
        propagated_signal_no_DGD_cuda = channel(signal)
        jones_matrices_no_DGD_cuda = channel.Jones(signal.frequency_angular)
        channel._polarisation_mode_dispersion = parameters.getfloat('FIBRE', 'polarisation_mode_dispersion')

        propagated_signal_no_DGD_cuda.to_device(Device.CPU)
        jones_matrices_no_DGD_cuda = jones_matrices_no_DGD_cuda.get()

        assert np.allclose(propagated_signal_no_DGD_cuda.samples_time, propagated_signal_no_DGD.samples_time), f"{type(channel)} CUDA- and CPU propagation yielded deviating results in the case without differential group delay"
        assert np.allclose(jones_matrices_no_DGD_cuda, jones_matrices_no_DGD), f"{type(channel)} CUDA- and CPU Jones matrices yielded deviating results in the case without differential group delay"
        
        signal.to_device(Device.CPU)
        
    # channel has type FibreMarcuse
    if 'cupy' in sys.modules: signal.to_device(Device.CUDA)

    # Test different perturbations
    perturbation_array = 1 + 10 * np.random.default_rng().normal(size = (channel.section_path.edge_count, 60))
    perturbation_material_strains = Perturbation(material_strains = perturbation_array, sample_rate = 1)
    perturbation_differential_phase_shifts = Perturbation(differential_phase_shifts = perturbation_array, sample_rate = 1)
    perturbation_twists = Perturbation(twists = perturbation_array, sample_rate = 1)

    propagated_signals_birefringence_scaled = channel(signal, transmission_start_times = [0, 30], perturbations = perturbation_material_strains)
    jones_matrices_birefringence_scaled = channel.Jones(signal.frequency_angular, transmission_start_times = [0, 30], perturbations = perturbation_material_strains)
    propagated_signals_birefringence_scaled.to_device(Device.CPU)
    jones_matrices_birefringence_scaled = jones_matrices_birefringence_scaled.get()

    propagated_signals_birefringence_added = channel(signal, perturbations = perturbation_differential_phase_shifts, transmission_start_times = [0, 30])
    jones_matrices_birefringence_added = channel.Jones(signal.frequency_angular, transmission_start_times = [0, 30], perturbations = perturbation_differential_phase_shifts)
    propagated_signals_birefringence_added.to_device(Device.CPU)
    jones_matrices_birefringence_added = jones_matrices_birefringence_added.get()

    propagated_signals_major_angles_added = channel(signal, perturbations = perturbation_twists, transmission_start_times = [0, 30])
    jones_matrices_major_angles_added = channel.Jones(signal.frequency_angular, transmission_start_times = [0, 30], perturbations = perturbation_twists)
    propagated_signals_major_angles_added.to_device(Device.CPU)
    jones_matrices_major_angles_added = jones_matrices_major_angles_added.get()

    assert np.allclose(propagated_signals_birefringence_scaled.power_W, signal.power_W), f"{type(channel)} birefringence scalar-perturbed fibre did not retain signal energy"
    assert np.allclose(propagated_signals_birefringence_added.power_W, signal.power_W), f"{type(channel)} birefringence adder-perturbed fibre did not retain signal energy"
    assert np.allclose(propagated_signals_major_angles_added.power_W, signal.power_W), f"{type(channel)} angle adder-perturbed fibre did not retain signal energy"
    
    assert not np.allclose(propagated_signals_birefringence_scaled.samples_time[:, 0, None], signal.samples_time), f"{type(channel)} birefringence scalar-perturbed fibre at 0 seconds matches input signal (but shouldn't)"
    assert not np.allclose(propagated_signals_birefringence_scaled.samples_time[:, 1, None], signal.samples_time), f"{type(channel)} birefringence scalar-perturbed fibre at 30 seconds matches input signal (but shouldn't)"
    assert not np.allclose(propagated_signals_birefringence_added.samples_time[:, 0, None], signal.samples_time), f"{type(channel)} birefringence adder-perturbed fibre at 0 seconds matches input signal (but shouldn't)"
    assert not np.allclose(propagated_signals_birefringence_added.samples_time[:, 1, None], signal.samples_time), f"{type(channel)} birefringence adder-perturbed fibre at 30 seconds matches input signal (but shouldn't)"
    assert not np.allclose(propagated_signals_major_angles_added.samples_time[:, 0, None], signal.samples_time), f"{type(channel)} angle adder-perturbed fibre at 0 seconds matches input signal (but shouldn't)"
    assert not np.allclose(propagated_signals_major_angles_added.samples_time[:, 1, None], signal.samples_time), f"{type(channel)} angle adder-perturbed fibre at 30 seconds matches input signal (but shouldn't)"

    assert not np.allclose(propagated_signals_birefringence_scaled.samples_time[:, 0, None], propagated_signal.samples_time), f"{type(channel)} birefringence scalar didn't perturb signal at 0 seconds"
    assert not np.allclose(propagated_signals_birefringence_scaled.samples_time[:, 1, None], propagated_signal.samples_time), f"{type(channel)} birefringence scalar didn't perturb signal at 30 seconds"
    assert not np.allclose(propagated_signals_birefringence_added.samples_time[:, 0, None], propagated_signal.samples_time), f"{type(channel)} birefringence adder didn't perturb signal at 0 seconds"
    assert not np.allclose(propagated_signals_birefringence_added.samples_time[:, 1, None], propagated_signal.samples_time), f"{type(channel)} birefringence adder didn't perturb signal at 30 seconds"
    assert not np.allclose(propagated_signals_major_angles_added.samples_time[:, 0, None], propagated_signal.samples_time), f"{type(channel)} angle adder didn't perturb signal at 0 seconds"
    assert not np.allclose(propagated_signals_major_angles_added.samples_time[:, 1, None], propagated_signal.samples_time), f"{type(channel)} angle adder didn't perturb signal at 30 seconds"
    
    assert not np.allclose(propagated_signals_birefringence_scaled.samples_time[:, 0, None], propagated_signals_birefringence_scaled.samples_time[:, 1, None]), f"{type(channel)} birefringence scalar perturbed signals at 0 and 30 seconds the same (but shouldn't have)"
    assert not np.allclose(propagated_signals_birefringence_added.samples_time[:, 0, None], propagated_signals_birefringence_added.samples_time[:, 1, None]), f"{type(channel)} birefringence adder perturbed signals at 0 and 30 seconds the same (but shouldn't have)"
    assert not np.allclose(propagated_signals_major_angles_added.samples_time[:, 0, None], propagated_signals_major_angles_added.samples_time[:, 1, None]), f"{type(channel)} angle adder perturbed signals at 0 and 30 seconds the same (but shouldn't have)"

    assert np.allclose(np.einsum('rbspq,rbsq->rbsp', jones_matrices_birefringence_scaled, signal.samples_frequency), propagated_signals_birefringence_scaled.samples_frequency), f"{type(channel)} birefringence scalar-perturbed fibre propagation and Jones matrix produced different results"
    assert np.allclose(np.einsum('rbspq,rbsq->rbsp', jones_matrices_birefringence_added, signal.samples_frequency), propagated_signals_birefringence_added.samples_frequency), f"{type(channel)} birefringence adder-perturbed fibre propagation and Jones matrix produced different results"
    assert np.allclose(np.einsum('rbspq,rbsq->rbsp', jones_matrices_major_angles_added, signal.samples_frequency), propagated_signals_major_angles_added.samples_frequency), f"{type(channel)} angle adder-perturbed fibre propagation and Jones matrix produced different results"

    assert not np.allclose(propagated_signals_birefringence_scaled.samples_frequency, propagated_signals_birefringence_added.samples_frequency), f"{type(channel)} birefringence scalar- and birefringence adder-perturbed fibre propagations produced the same results (but shouldn't have)"
    assert not np.allclose(propagated_signals_birefringence_scaled.samples_frequency, propagated_signals_major_angles_added.samples_frequency), f"{type(channel)} birefringence scalar- and angle adder-perturbed fibre propagations produced the same results (but shouldn't have)"
    assert not np.allclose(propagated_signals_birefringence_added.samples_frequency, propagated_signals_major_angles_added.samples_frequency), f"{type(channel)} birefringence adder- and angle adder-perturbed fibre propagations produced the same results (but shouldn't have)"

def test_fibre_initialisation():
    for modulus_model in 'RANDOM', 'FIXED':
        parameters.set('FIBRE', 'modulus_model', modulus_model)
        for channel in (FibreMarcuse(parameters), FibreCoarseStep(parameters)):
            differential_group_delay = channel.differential_group_delay
            assert np.isclose(np.mean(differential_group_delay), channel.polarisation_mode_dispersion * np.sqrt(channel.length), rtol = 2e-1), f"Accumulated differential group delay does not match polarisation mode dispersion parameter"

            if isinstance(channel, FibreMarcuse): continue
            # # Check if accumulated differential group delay is Maxwellian-distributed.
            # # Curti et al. - Statistical Treatment of the Evolution of the Principal States of Polarization in Single-Mode Fibers
            # assert sp.stats.kstest(differential_group_delay, lambda x: sp.stats.maxwell.cdf(x, scale = channel.polarisation_mode_dispersion * np.sqrt(channel.length * np.pi / 8))).pvalue > 0.05, "Accumulated differential group delay does not approach correct Maxwell-Bolzmann distribution"

            # Check scramblers in the coarse-step model
            assert np.allclose(channel.scramblers.swapaxes(-2, -1).conjugate() @ channel.scramblers, np.eye(2)), f"Fibre scramblers are not unitary"

def test_fibre_path():
    for channel in (FibreMarcuse(parameters_geographic), FibreCoarseStep(parameters_geographic)):
        assert np.allclose(channel.section_path.lengths[:-1], parameters_geographic.getfloat('FIBRE', 'section_length')), "Fibre section lengths didn't match parameter section_length with geographic initialisation"
        assert np.isclose(np.sum(channel.section_path.lengths), np.sum(channel.path.lengths)), f"Fibre path- and section lengths add up to {np.sum(channel.path.lengths)} and {np.sum(channel.section_path.lengths)}, but should have the same total"

        try:
            channel.path.coordinates
            channel.section_path.coordinates
        except AttributeError:
            raise AssertionError("Channel should contain a path and a section_path with coordinates, but didn't")

        transmitter = Transmitter(parameters)
        _, signal = transmitter.transmit_random(1, int(parameters.getfloat("SIGNAL", "symbol_count")))

        if 'cupy' in sys.modules: signal.to_device(Device.CUDA)

        propagated_signal = channel(signal)
        assert signal.xp.allclose(signal.power_W, propagated_signal.power_W), f"Fibre did not retain signal energy"
        assert not signal.xp.allclose(signal.samples_time, propagated_signal.samples_time), f"Fibre output matched the input (but shouldn't)"
        jones_matrices = channel.Jones(signal.frequency_angular)
        assert signal.xp.allclose(signal.xp.einsum('rbspq,rbsq->rbsp', jones_matrices, signal.samples_frequency), propagated_signal.samples_frequency), f"Fibre propagation and Jones matrix produced different results"


def test_fibre_dict():
    for channel in (FibreMarcuse(parameters), FibreCoarseStep(parameters)):
        fibre_dict = channel.to_dict()
        channel_copy = type(channel).from_dict(fibre_dict)
        assert channel == channel_copy, f"Channel was not copied properly using to_dict and from_dict"

    for channel in (FibreMarcuse(parameters_geographic), FibreCoarseStep(parameters_geographic)):
        fibre_dict = channel.to_dict()
        channel_copy = type(channel).from_dict(fibre_dict)
        assert channel == channel_copy, f"Geographic channel was not copied propertly using to_dict and from_dict"