"""
An optical fibre channel model base class for dual-polarisation transmission.
"""

from configparser import ConfigParser
import json
import logging
import sys
from abc import ABC, abstractmethod

from tqdm import tqdm
import numpy as np
import scipy as sp
try:
    import cupy as cp
except:
    pass
import obspy as op
import refractiveindex

from .constants import Device, Domain, ModulusModel, Gain
from .utilities import dB2linear
from .signal import Signal
from .perturbation import Perturbation
from .path import Path

logger = logging.getLogger()

class Fibre(ABC):
    """
    A base class representing an optical fibre.
    Currently it models only polarisation-mode dispersion using one of two methods:
    - the coarse-step method, which applies differential group delay and scrambles the state of polarisation in a random and distributed manner.
    - Marcuse's method, which was derived directly from the coupled nonlinear Schrödinger equation.
    Chromatic dispersion, the Kerr effect, attenuation, EDFA noise, and polarisation-dependent loss are neglected, and slow polarisation mode dispersion drift are not implemented.
    External perturbations can be modelled as a change in differential phase or major birefringence axes orientations.
    """
    def __init__(self, parameters: ConfigParser):
        """
        Instantiate multiple fibre channels for simultaneous propagation.

        Required and optional entries in parameters['FIBRE']:
        - correlation_length [float]:           Correlation length in km.
        - beat_length [float]:                  Beat length in km.
        - span_length [float]:                  Fibre span length in km, after which an amplifier is added.
        - steps_per_span [int]:                 Number of split-step steps per fibre span.
        - path_coordinates [list]:              (Optional) If section_count is not defined, list of coordinates (longitude, latitude) along the fibre path.
        - span_count [int]:                     (Optional) If path_coordinates is not defined, number of fibre spans.
        - chromatic_dispersion [float]:         Chromatic dispersion parameter in ps^2/km.
        - nonlinearity [float]:                 Nonlinearity parameter in  1 / (W km).
        - attenuation [float]:                  Attenuation in dB / km.
        - noise_figure [float]:                 Amplifier noise figure in dB.
        - polarisation_mode_dispersion [float]: Average accumulated differential group delay in ps/(km ^ 0.5); If 0, turns off major axes rotations as well.
        - realisation_count [int]:              Number of fibre realisations with different distributed polarisation mode dispersion.
        - photoelasticity [float]:              Fibre photoelasticity.
        - modulus_model [str]:                  Method (FIXED or RANDOM) to generate polarisation mode dispersion realisations.
        """
        logger.info("Creating fibre..")

        assert 'FIBRE'                        in parameters, f"Parameters are missing section 'FIBRE'."
        for field in (
                'correlation_length',
                'beat_length',
                'span_length',
                'steps_per_span',
                'chromatic_dispersion',
                'nonlinearity',
                'attenuation',
                'noise_figure',
                'polarisation_mode_dispersion',
                'realisation_count',
                'photoelasticity',
                'modulus_model'
            ):
            assert field in parameters['FIBRE'], f"'{field}' is missing from parameters section 'FIBRE'."
        
        self._correlation_length           = parameters.getfloat('FIBRE', 'correlation_length')
        self._beat_length                  = parameters.getfloat('FIBRE', 'beat_length')
        self._span_length                  = parameters.getfloat('FIBRE', 'span_length')
        self._steps_per_span               = parameters.getint('FIBRE', 'steps_per_span')
        self._chromatic_dispersion         = parameters.getfloat('FIBRE', 'chromatic_dispersion')
        self._nonlinearity                 = parameters.getfloat('FIBRE', 'nonlinearity')
        self._attenuation_dB               = parameters.getfloat('FIBRE', 'attenuation')
        self._noise_figure_dB              = parameters.getfloat('FIBRE', 'noise_figure')
        self._polarisation_mode_dispersion = parameters.getfloat('FIBRE', 'polarisation_mode_dispersion')
        self._realisation_count            = int(parameters.getfloat('FIBRE', 'realisation_count'))
        self._photoelasticity              = parameters.getfloat('FIBRE', 'photoelasticity')
        self._modulus_model                = ModulusModel[parameters.get('FIBRE', 'modulus_model')]
        self._material                     = refractiveindex.RefractiveIndexMaterial(
            shelf = 'glass',
            book  = 'fused_silica',
            page  = 'Malitson'
        )

        for field in (
                '_correlation_length',
                '_beat_length',
                '_span_length',
                '_steps_per_span',
                '_realisation_count',
            ):
            assert getattr(self, field) > 0, f"{field} must be >0, but was {field}"
            
        for field in (
                '_attenuation_dB',
                '_noise_figure_dB',
            ):
            assert getattr(self, field) >= 0, f"{field} must be >=0, but was {field}"      
        
        if 'path_coordinates' in parameters['FIBRE']:
            self._path = Path(
                    *np.array(json.loads(parameters.get('FIBRE', 'path_coordinates')), dtype = float).T
                )
            self._span_count = None

        else:
            self._path       = None
            self._span_count = parameters.getint('FIBRE', 'span_count')

        self._init_path()
        self._init_birefringence()

    def _init_path(self):
        """
        Initialise fibre- and step path information.
        """
        logger.info("Generating fibre path..")
        if self._path is not None:
            self._span_path = self.path.interpolated(self._span_length)

        else:
            self._span_path = Path(lengths = np.full(
                    shape      = (self._span_count,),
                    fill_value = self._span_length,
                    dtype      = float
                ))

        step_coordinates_or_lengths = []
        step_gains_dB = []
        step_length = self._span_length / self.steps_per_span

        for span in self.span_path:
            span_step_path = span.interpolated(step_length)
            try:
                step_coordinates_or_lengths.append(span_step_path.coordinates[:-1])
            except:
                step_coordinates_or_lengths.append(span_step_path.lengths)
        
            step_gains_dB.extend([0,] * (span_step_path.edge_count - 1) + [self.attenuation_dB * span.length])

        try:
            step_coordinates_or_lengths.append(span_step_path.coordinates[-1, None])
            step_coordinates = np.concatenate(step_coordinates_or_lengths, axis = 0)
            self._step_path = Path(*step_coordinates.T)
        except:
            step_lengths = np.concatenate(step_coordinates_or_lengths, axis = 0)
            self._step_path = Path(lengths = step_lengths)
        
        self._step_gains_dB = np.array(step_gains_dB)

    def _init_birefringence(self):
        """
        Generate differential phase shifts, group delays, and major axes orientations or scramblers.
        """
        if self.polarisation_mode_dispersion != 0.:
            logger.info("Generating fibre birefringence realisations..")
            match self.modulus_model:
                case ModulusModel.FIXED:  self._init_birefringence_fixed()
                case ModulusModel.RANDOM: self._init_birefringence_random()
                case _: raise AssertionError(f"modulus_model must be ModulusModel.FIXED or ModulusModel.RANDOM, but was {self.modulus_model}")

    @abstractmethod
    def _init_birefringence_fixed(self):
        """
        Initialise birefringences with equal moduli, e.g. using the fixed modulus model.
        """

    @abstractmethod
    def _init_birefringence_random(self):
        """
        Initialise birefringences with random moduli, e.g. using the random modulus model.
        """

    def __call__(self, signal: Signal, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = [], step_start: int = None, step_stop: int = None) -> Signal:
        """
        Make fibre instances callable; see propagate()
        """
        return self.propagate(signal, transmission_start_times, perturbations, step_start, step_stop)

    def propagate(self, signal: Signal, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = [], step_start: int = None, step_stop: int = None) -> Signal:
        """
        Method called by propagate() and Jones() to simulate the fibre response.

        Inputs:
        - signal [Signal]: the signal or transposed Jones matrix to propagate through the channel, shape [R,B,S,P] with number of realisations R or R = 1, batch size B, sample count S and principal polarisations P = 2, OR shape [R, S, P, P] where the last two axes contain Jones transfer matrices.
        - transmission_start_times [float, np.ndarray]: timestamp(s) at which the signal transmission(s) begins in s, relative to the start time of a perturbation. If not a float, shape [T,].
        - perturbations [Perturbation, list]: Model these perturbations during signal transmission, in order of appearance. All birefringence scaling is applied before addition.
        - step_start [int]: Index of the first fibre step to model. If None, defaults to 0.
        - step_stop [int]: Index of the first fibre step not to model. If None, defaults to self.step_path.edge_count

        Outputs:
        - [Signal]: the output signal or transposed Jones matrix, same shape as signal
        """
        if step_start is not None:
            assert isinstance(step_start, (int, np.integer)), f"step_start must be an int, but was a {type(step_start)}"
            assert step_start >= 0, f"step_start must be >= 0, but was {step_start}"
            assert step_start < self.step_path.edge_count, f"step_start must be < self.step_path.edge_count ({self.step_path.edge_count}), but was {step_start}"

        if step_stop is not None:
            assert isinstance(step_stop, (int, np.integer)), f"step_stop must be an int, but was a {type(step_stop)}"
            assert step_start is None or step_stop > step_start, f"step_stop must be > step_start ({step_start}), but was {step_stop}"
            if step_stop > self.step_path.edge_count:
                logger.warning(f"step_stop {step_stop} is larger than the number of fibre steps {self.step_path.edge_count}")

        signal                   = self._prepare_signal(signal)
        transmission_start_times = self._prepare_transmission_start_times(signal, transmission_start_times)
        perturbations            = self._prepare_perturbations(signal, perturbations)
        steps_iterable           = self._prepare_steps_iterable(signal, perturbations, step_start, step_stop)
        group_velocity           = self._prepare_group_velocity(signal)
        noise_figure_linear      = self.noise_figure_linear # Prevent recalculation at every step
        
        linear_exponent = signal.xp.zeros(shape = (self.realisation_count, max(len(transmission_start_times), signal.shape[1]), *signal.shape[2:]), dtype = complex) # [R, B, S, P]

        for step_index, step_values in enumerate(steps_iterable):
            step_length = step_values[0]
            step_gain_linear = step_values[1]
            step_centre_position = step_values[2]
            birefringence_quantities = step_values[3:]
            linear_exponent[:] = 0

            if self.chromatic_dispersion != 0.:
                linear_exponent = self._apply_chromatic_dispersion(linear_exponent, signal, step_length / 2)

            if self.polarisation_mode_dispersion != 0.:
                if len(perturbations):
                    perturbations_sample_masks, perturbations_sample_indices = self._step_perturbations_indices(signal, perturbations, transmission_start_times, step_centre_position, group_velocity)
                    birefringence_quantities = self._perturb_birefringence_quantities(signal, step_index, perturbations, perturbations_sample_masks, perturbations_sample_indices, *birefringence_quantities)
                
                signal = self._prepare_birefringence(signal, *birefringence_quantities)
                linear_exponent = self._apply_half_birefringence(linear_exponent, signal, *birefringence_quantities)

            if self.nonlinearity != 0.:
                signal = self._apply_linear_exponent(linear_exponent, signal) # Apply half the linear effects in the frequency domain
                linear_exponent[:] = 0 # Re-initialise exponent to zeros
                signal = self._apply_nonlinearity(signal, step_length) # Apply the nonlinear effects in the time domain

            if self.attenuation_dB != 0.:           linear_exponent = self._apply_attenuation(linear_exponent, step_length)
            if self.chromatic_dispersion != 0.:         linear_exponent = self._apply_chromatic_dispersion(linear_exponent, signal, step_length / 2)
            if self.polarisation_mode_dispersion != 0.: linear_exponent = self._apply_half_birefringence(linear_exponent, signal, *birefringence_quantities)

            signal = self._apply_linear_exponent(linear_exponent, signal) # Apply the remaining linear effects in the frequency domain

            if self.polarisation_mode_dispersion != 0.:
                signal = self._finalise_birefringence(signal, *birefringence_quantities)

            if step_gain_linear != 1.:
                signal = self._apply_gain(signal, step_gain_linear, noise_figure_linear)

        return signal

    def Jones(self, sample_rate: float = None, sample_count: int = 1, carrier_wavelength: float = 1550., device: Device = Device.CPU, signal: Signal = None, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = [], step_start: int = None, step_stop: int = None) -> np.ndarray:
        """
        Calculate the fibre Jones matrix.
        The matrix will exclude any noise or nonlinear effects.
        Internally, this function builds a frequency-dependent signal and calls propagate(), after which a frequency-dependent Jones matrix is constructed from the result.

        Inputs:
        - sample_rate [float]: sample rate in samples per second of the internally used Signal.
        - sample_count [int]: number of frequencies at which to evaluate the Jones matrix. This will result in sample_count frequencies uniformly spaced in the open interval [-sample_rate/2, sample_rate/2)
        - carrier_wavelength [float]: carrier wavelength in nm
        - device [Device]: Device on which to perform the calculations.
        - signal [Signal]: if not None, use the sample_rate, sample_count, carrier_wavelength and device from this signal instead
        - transmission_start_times [float, np.ndarray]: timestamp(s) at which the signal transmission(s) begins in s, relative to the start time of a perturbation. If not a float, shape [T,].
        - perturbations [Perturbation, list]: Model these perturbations during signal transmission, in order of appearance. All birefringence scaling is applied before addition.
        - step_start [int]: Index of the first fibre step to model. If None, defaults to 0.
        - step_stop [int]: Index of the first fibre step not to model. If None, defaults to self.step_path.edge_count
        
        Outputs:
        - [Signal]: the Jones matrices, shape [R,T,S,2,2] where R is the fibre realisation count, S the sample_count, and the last two axes contain the matrices
        """
        assert self.nonlinearity == 0., f"Jones matrices can only be calculated for linear fibres, but the nonlinearity coefficient was {self.nonlinearity} != 0."

        if signal is not None:
            sample_rate        = signal.sample_rate
            sample_count       = signal.sample_count
            carrier_wavelength = signal.carrier_wavelength
            device = signal.device

        assert sample_count == 1 or (sample_rate is not None and sample_rate > 0), f"sample_rate must be > 0, but was {sample_rate}"
        assert sample_count >= 1, f"sample_count must be >= 1, but was {sample_count}"

        probe_samples_frequency = np.full(shape = (1, 1, sample_count, 2), fill_value = [1, 0], dtype = complex)
        probe_signal = Signal(
            samples = probe_samples_frequency,
            sample_rate = sample_rate,
            domain = Domain.FREQUENCY,
            carrier_wavelength = carrier_wavelength
        )
        probe_signal.to_device(device)

        probed_signal = self.propagate(probe_signal, transmission_start_times, perturbations, step_start, step_stop)

        Jones_columns_1 = probed_signal.samples_frequency
        Jones_columns_2 = probed_signal.samples_frequency
        Jones_columns_2 = probed_signal.xp.flip(Jones_columns_2, axis = -1)
        Jones_columns_2 = probed_signal.xp.conjugate(Jones_columns_2)
        Jones_columns_2[..., 0] = -Jones_columns_2[..., 0]
        Jones_matrices = Signal(
            samples = probed_signal.xp.stack([Jones_columns_1, Jones_columns_2], axis = -1),
            sample_rate = sample_rate,
            sample_axis = -3,
            domain = Domain.FREQUENCY,
            carrier_wavelength = carrier_wavelength
        )

        return Jones_matrices

    def accumulate_differential_group_delays(self, device: Device = Device.CPU, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = [], step_start: int = None, step_stop: int = None) -> np.ndarray:
        """
        Accumulated differential group delay in ps, shape [R] where R is the number of fibre realisations.
        See Jones() for an explanation of the arguments.
        """
        match(device):
            case Device.CPU: xp = np
            case Device.CUDA:
                assert 'cupy' in sys.modules, f"Cannot calculate differential group delay on CUDA; no cupy installation found"
                xp = cp
                
        Jones_matrices = self.Jones(sample_rate = xp.pi / (30 * self.step_path.lengths[0]), sample_count = 2, device = device, transmission_start_times = transmission_start_times, perturbations = perturbations, step_start = step_start, step_stop = step_stop)
        Jones_matrices_derivative = xp.diff(Jones_matrices.samples_frequency, axis = 2)[:, 0] / (-2 * xp.pi * Jones_matrices.sample_rate / Jones_matrices.sample_count) #xp.diff(Jones_matrices.frequency_angular)[0]

        differential_group_delay = 2 * xp.sqrt(xp.linalg.det(Jones_matrices_derivative)) * 1e12 # Gordon et al. - PMD Fundamentals: Polarization Mode Dispersion in Optical Fibres
        differential_group_delay = differential_group_delay.real.astype(float)

        if xp != np: differential_group_delay = differential_group_delay.get()

        return differential_group_delay

    def _prepare_signal(self, signal):
        assert signal.sample_axis_negative == -2, f"signal must have sample axis -2, but it was {signal.sample_axis_negative}"
        assert signal.shape[-1] == 2, f"signal must have two polarisations on the last axis, but had {signal.shape[-1]}"
        assert len(signal.shape) == 4, f"signal must have shape [R,B,S,P], but had shape {signal.shape}"
        signal = signal.copy()
        if signal.device == Device.CUDA: signal.to_device(signal.device) # Ensure that signal resides in the currently active GPU, if any
        return signal

    def _prepare_perturbations(self, signal, perturbations):
        if not isinstance(perturbations, (list, tuple)):
            perturbations = (perturbations,)

        for perturbation in perturbations:
            assert isinstance(perturbation, Perturbation), f"All perturbations must have type Perturbation, but at least one had type {type(perturbation)}"
            perturbation.to_device(signal.device)

        return perturbations
        
    def _prepare_transmission_start_times(self, signal, transmission_start_times):
        if isinstance(transmission_start_times, (int, np.integer, float, np.floating)):
            transmission_start_times = [transmission_start_times]
        
        transmission_start_times = signal.invite_array(transmission_start_times)

        assert len(transmission_start_times.shape) == 1, f"transmission_start_times must have shape [T,], but had shape {transmission_start_times.shape}"
        if len(transmission_start_times > 1):
            assert signal.shape[1] in (1, len(transmission_start_times)), f"If transmission_start_times has shape [T > 1,] signal must have batch size 1 or T ({len(transmission_start_times)}), but this was {signal.shape[1]}"
        
        return transmission_start_times

    def _prepare_steps_iterable(self, signal, perturbations, step_start, step_stop):
        steps_iterable_arrays = self._prepare_steps_iterable_arrays(signal, step_start, step_stop)
        steps_iterable = zip(*steps_iterable_arrays)

        if step_start is not None:
            step_start = max(step_start, 0)

        if step_stop is not None:
            step_stop = min(step_stop, self.step_path.edge_count)

        desc_string = "Propagating signal "
        if step_start is not None: desc_string += f"from fibre step {step_start + 1} "
        if step_stop is not None: desc_string += f"until step {step_stop} of {self.step_path.edge_count} "
        desc_string += "("
        desc_string += "CPU" if signal.device == Device.CPU else "CUDA"
        if len(perturbations): desc_string += ", perturbed"
        desc_string += ")"
        if logger.isEnabledFor(logging.INFO):
            steps_iterable = tqdm(
                steps_iterable,
                total = (self.step_path.edge_count if step_stop is None else step_stop) - (0 if step_start is None else step_start),
                desc = desc_string
            )

        return steps_iterable

    def _prepare_group_velocity(self, signal):
        return signal.invite_array(self.group_velocity(signal.carrier_wavelength))

    @abstractmethod
    def _prepare_steps_iterable_arrays(self, signal, step_start, step_stop):
        step_lengths              = signal.invite_array(self.step_path.lengths[step_start:step_stop])
        step_gains_linear         = signal.invite_array(self.step_gains_linear[step_start:step_stop])
        step_centre_positions     = signal.invite_array(self.step_path.centre_positions[step_start:step_stop])
        differential_group_delays = signal.invite_array(self.differential_group_delays[step_start:step_stop, :, None, None]) # [S, R, 1, 1]

        return step_lengths, step_gains_linear, step_centre_positions, differential_group_delays

    def _step_perturbations_indices(self, signal, perturbations, transmission_start_times, step_centre_position, group_velocity):
        perturbations_sample_times      = transmission_start_times + step_centre_position / group_velocity # [B,]
        perturbations_sample_masks      = signal.xp.zeros(shape = (len(perturbations), len(perturbations_sample_times)), dtype = bool) # [P, B]
        perturbations_sample_indices    = signal.xp.zeros_like(perturbations_sample_masks, dtype = int) # [P, B]
        for index, perturbation in enumerate(perturbations):
            perturbations_sample_masks[index]      = (perturbations_sample_times >= perturbation.start_time) & (perturbations_sample_times < perturbation.start_time + perturbation.duration)
            perturbations_sample_indices[index]    = signal.xp.floor((perturbations_sample_times - perturbation.start_time) * perturbation.sample_rate).astype(int)
        
        return perturbations_sample_masks, perturbations_sample_indices

    @abstractmethod
    def _perturb_birefringence_quantities(self, signal, step_index, perturbations, perturbation_sample_masks, perturbation_sample_indices, *birefringence_quantities):
        pass

    def _apply_chromatic_dispersion(self, linear_exponent, signal, step_length):
        # signal.samples_frequency = signal.samples_frequency * signal.xp.exp(1j * 0.5 * self.chromatic_dispersion * signal.frequency_angular ** 2 * step_length * 1e-24)[None, None, :, None]
        # return signal
        return linear_exponent + 0.5j * self.chromatic_dispersion * signal.frequency_angular[None, None, :, None] ** 2 * step_length * 1e-24 # [R, B, S, P]

    @abstractmethod
    def _prepare_birefringence(self, signal, birefringence_quantities):
        pass

    @abstractmethod
    def _apply_half_birefringence(self, linear_exponent, signal, *birefringence_quantities):
        pass

    @abstractmethod
    def _finalise_birefringence(self, linear_exponent, signal, birefringence_quantities):
        pass

    def _apply_nonlinearity(self, signal, step_length):
        """
        Nonlinearity is applied only when propagating a signal (not when building a Jones matrix). Therefore, signal always has shape [R, B, S, P] here with realisations R, batch size B, time/frequency axis S and polarisations P = 2
        """
        # signal.samples_time = signal.samples_time * signal.xp.exp(1j * 8 / 9 * self.nonlinearity * step_length * np.linalg.norm(signal.samples_time, axis = -1)[:, :, :, None] ** 2) # ??? Wrong for now, look at Marcuse's paper
        logger.error("Fibre nonlinearity is an ad-hoc implementation and cannot be assumed to be correct")

        signal_power   = signal.xp.linalg.norm(signal.samples_time, axis = -1)[:, :, :, None, None] ** 2 # [R, B, S, 1, 1]

        signal_rotator = -1/3 * signal.xp.einsum(
            'rbsp,pq,rbsq->rbs',
            signal.samples_time,
            PAULI_3 if signal.device == Device.CPU else PAULI_3_CUDA,
            signal.samples_time,
            optimize = True
        )[:, :, :, None, None] * (PAULI_3 if signal.device == Device.CPU else PAULI_3_CUDA) # [R, B, S, 2, 2]
        
        nonlinearity_operator = signal.xp.exp(1j * self.nonlinearity * (signal_power + signal_rotator) * step_length) # Eq. 2 in Marcuse et al. (1997), Eq. 5 in Menyuk et al. (1987). ??? halve kerr parameter as in Menyuk et al?
        nonlinearity_operator = signal.xp.moveaxis(nonlinearity_operator, (-1, -2)) # Transpose operator for efficient einsum
        
        signal.samples_time = signal.xp.einsum(
            'rbsqp,rbsq->rbsp',
            nonlinearity_operator,
            signal.samples_time,
            optimize = True
        )

        return signal

    def _apply_attenuation(self, linear_exponent, step_length):
        """
        Attenuation is applied only when propagating a signal (not when building a Jones matrix). Therefore, signal always has shape [R, B, S, P] here with realisations R, batch size B, time/frequency axis S and polarisations P = 2
        """
        return linear_exponent - self.attenuation_natural * step_length

    def _apply_linear_exponent(self, linear_exponent, signal):
        signal.samples_frequency = signal.samples_frequency * signal.xp.exp(linear_exponent)
        return signal

    def _apply_gain(self, signal, gain_linear, noise_figure_linear):
        noise_power_per_channel = noise_figure_linear * gain_linear * sp.constants.Planck * signal.carrier_frequency * signal.bandwidth / 2 / 2 # /2 to divide over polarisations, /2 to spread over perpendicular phases. Don't scale by sqrt(sample_count), because the fourier transforms used internally in Signal are orthonormal
        # noise_power_per_channel = (noise_figure_linear * gain_linear - 1) * sp.constants.Planck * signal.carrier_frequency * signal.bandwidth / 2 / 2
        
        if signal.xp == np:
            amplified_spontaneous_emission = np.sqrt(noise_power_per_channel) * (np.random.default_rng().normal(size = signal.shape) + 1j * np.random.default_rng().normal(size = signal.shape))
        else:
            amplified_spontaneous_emission = cp.sqrt(noise_power_per_channel) * (cp.random.normal(size = signal.shape) + 1j * cp.random.normal(size = signal.shape))
        
        # signal.samples += amplified_spontaneous_emission / 2
        # signal.samples *= gain_linear
        # signal.samples += amplified_spontaneous_emission / 2
        signal.samples = signal.samples * gain_linear + amplified_spontaneous_emission

        return signal

    def group_velocity(self, carrier_wavelength: float):
        """
        Obtain the fibre propagation constant (inverse of group velocity) for a specific signal in km/s

        Inputs:
        - carrier_wavelength [float]: the carrier wavelength at which to obtain the propagation constant

        Outputs:
        - [float] group velocity in km/s
        """
        return sp.constants.speed_of_light / self.material.get_refractive_index(carrier_wavelength) / 1000

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Represent this fibre (and its exact realisations) as a dictionary.

        Outputs:
        - [dict] The dictionary representation of this fibre
        """
        fibre_dict = {
            'correlation_length':           self.correlation_length,
            'beat_length':                  self.beat_length,
            'span_path':                    self.span_path.to_dict(),
            'step_path':                    self.step_path.to_dict(),
            'steps_per_span':               self.steps_per_span,
            'step_gains':                   self.step_gains_dB.tolist(),
            'chromatic_dispersion':         self.chromatic_dispersion,
            'nonlinearity':                 self.nonlinearity,
            'attenuation':                  self.attenuation_dB,
            'noise_figure':                 self.noise_figure_dB,
            'polarisation_mode_dispersion': self.polarisation_mode_dispersion,
            'realisation_count':            self.realisation_count,
            'photoelasticity':              self.photoelasticity,
            'modulus_model':                self.modulus_model.name,
            'differential_group_delays':    self.differential_group_delays.tolist()
        }

        if self._path is not None:
            fibre_dict = fibre_dict | {
                'path': self.path.to_dict()
            }

        return fibre_dict

    @classmethod
    @abstractmethod
    def from_dict(cls, fibre_dict: dict):
        """
        Instantiate a fibre from a saved dictionary.

        Inputs:
        - fibre_dict [dict]: a dictionary created using Fibre.to_dict()

        Outputs:
        - [Fibre] the loaded fibre instance.
        """
        parameters = ConfigParser()
        parameters.add_section('FIBRE')
        parameters.set('FIBRE', 'correlation_length',           str(fibre_dict['correlation_length']))
        parameters.set('FIBRE', 'beat_length',                  str(fibre_dict['beat_length']))
        parameters.set('FIBRE', 'steps_per_span',               str(fibre_dict['steps_per_span']))
        parameters.set('FIBRE', 'chromatic_dispersion',         str(fibre_dict['chromatic_dispersion']))
        parameters.set('FIBRE', 'nonlinearity',                 str(fibre_dict['nonlinearity']))
        parameters.set('FIBRE', 'attenuation',                  str(fibre_dict['attenuation']))
        parameters.set('FIBRE', 'noise_figure',                 str(fibre_dict['noise_figure']))
        parameters.set('FIBRE', 'polarisation_mode_dispersion', str(fibre_dict['polarisation_mode_dispersion']))
        parameters.set('FIBRE', 'realisation_count',            str(fibre_dict['realisation_count']))
        parameters.set('FIBRE', 'photoelasticity',              str(fibre_dict['photoelasticity']))
        parameters.set('FIBRE', 'modulus_model',                fibre_dict['modulus_model'])

        span_path = Path.from_dict(fibre_dict['span_path'])
        parameters.set('FIBRE', 'span_length', str(span_path.lengths[0]))
        
        if 'path' in fibre_dict:
            path = Path.from_dict(fibre_dict['path'])
            parameters.set('FIBRE', 'path_coordinates', json.dumps(path.coordinates.tolist()))
        else:
            parameters.set('FIBRE', 'span_count', str(span_path.edge_count))

        fibre = cls(parameters)
        fibre.differential_group_delays = np.array(fibre_dict['differential_group_delays'])
        
        fibre._step_path = Path.from_dict(fibre_dict['step_path'])
        fibre._step_gains_dB = np.array(fibre_dict['step_gains'])
        fibre._span_path = span_path
        if 'path' in fibre_dict:
            fibre._path = path

        return fibre

    def __eq__(self, other) -> bool:
        return self._polarisation_mode_dispersion  == other._polarisation_mode_dispersion and \
            self._photoelasticity                  == other._photoelasticity              and \
            self._chromatic_dispersion             == other._chromatic_dispersion         and \
            self._nonlinearity                     == other._nonlinearity                 and \
            self._attenuation_dB                   == other._attenuation_dB               and \
            self._noise_figure_dB                  == other._noise_figure_dB              and \
            self._realisation_count                == other._realisation_count            and \
            self._step_path                        == other._step_path                    and \
            self._span_path                        == other._span_path                    and \
            self._path                             == other._path                         and \
            np.all(self._step_gains_dB             == other._step_gains_dB)               and \
            np.all(self._differential_group_delays == other._differential_group_delays)   and \
            self._modulus_model                    == other._modulus_model

    @property
    def path(self) -> Path:
        """
        [Path] segmented fibre path
        """
        if self._path is None:
            raise AttributeError("Path coordinates were not passed to Fibre constructor, and are therefore not available")
        return self._path

    @path.setter
    def path(self, value):
        raise AttributeError("The path cannot be changed after instantiation of the Fibre")

    @property
    def span_path(self) -> Path:
        """
        [Path] segmented fibre path, divided into spans which end with an amplifier. May or may not include earth coordinates, depending on how the fibre was created
        """
        return self._span_path

    @span_path.setter
    def span_path(self, value):
        raise AttributeError("The span path cannot be changed after instantiation of the Fibre")

    @property
    def steps_per_span(self) -> int:
        """
        [int] number of split-step simulation steps per fibre span
        """
        return self._steps_per_span

    @steps_per_span.setter
    def steps_per_span(self, value):
        raise AttributeError("The steps per span cannot be changed after instantiation of the Fibre")

    @property
    def step_path(self) -> Path:
        """
        [Path] segmented fibre path, divided into short split-step steps. May or may not include earth coordinates, depending on the fibre parameters
        """
        return self._step_path

    @step_path.setter
    def step_path(self, value):
        raise AttributeError("The step path cannot be changed after instantiation of the Fibre")

    @property
    def step_gains_dB(self) -> list:
        """
        [list] for each step in step_path, contains a gain in dB if this step has an amplifier at the end, and otherwise 0.
        """
        return self._step_gains_dB

    @step_gains_dB.setter
    def step_gains_dB(self, value):
        raise AttributeError("The gain per step cannot be changed after instantiation of the Fibre")

    @property
    def step_gains_linear(self) -> list:
        """
        [list] for each step in step_path, contains an gain scalar if this step has an amplifier at the end, and otherwise 0.
        """
        return dB2linear(self.step_gains_dB, Gain.AMPLITUDE)

    @step_gains_linear.setter
    def step_gains_linear(self, value):
        raise AttributeError("The gain per step cannot be changed after instantiation of the Fibre")

    @property
    def correlation_length(self) -> float:
        """
        [float] the fibre correlation length in km, after which the state of polarisation is uncorrelated to its initial state
        """
        return self._correlation_length

    @correlation_length.setter
    def correlation_length(self, value):
        raise AttributeError("The correlation length correlation_length cannot be changed after instantiation of the Fibre")

    @property
    def beat_length(self) -> float:
        """
        [float] the fibre beat length in km, in which the differential phase rotates 2pi radians
        """
        return self._beat_length

    @beat_length.setter
    def beat_length(self, value):
        raise AttributeError("The beat length beat_length cannot be changed after instantiation of the Fibre")

    @property
    def chromatic_dispersion(self) -> float:
        """
        [float] chromatic dispersion parameter of this Fibre in ps ^ 2 / km / rad
        """
        return float(self._chromatic_dispersion)

    @chromatic_dispersion.setter
    def chromatic_dispersion(self, value):
        raise AttributeError("The chromatic dispersion parameter chromatic_dispersion cannot be changed after instantiation of the Fibre")

    @property
    def nonlinearity(self) -> float:
        """
        [float] nonlinearity parameter of this Fibre in rad / (W km)
        """
        return float(self._nonlinearity)

    @nonlinearity.setter
    def nonlinearity(self, value):
        raise AttributeError("The nonlinearity parameter cannot be changed after instantiation of the Fibre")

    @property
    def attenuation_dB(self) -> float:
        """
        [float] attenuation of this Fibre in dB / km
        """
        return float(self._attenuation_dB)

    @attenuation_dB.setter
    def attenuation_dB(self, value):
        raise AttributeError("The attenuation parameter cannot be changed after instantiation of the Fibre")

    @property
    def attenuation_natural(self) -> float:
        """
        [float] the exponent that applies attenuation of this Fibre per km. exp(attenuation_natural * step_length) = 10 ** (-attenuation_dB * step_length / 20)
        """
        return float(self.attenuation_dB * 0.115129255) # 0.115129255 = ln(10) / 20

    @attenuation_natural.setter
    def attenuation_natural(self) -> float:
        raise AttributeError("attenuation_natural cannot be changed after instantiation of the Fibre")

    @property
    def noise_figure_dB(self):
        """
        [float] the amplifier noise figure in dB.
        """
        return self._noise_figure_dB

    @noise_figure_dB.setter
    def noise_figure_dB(self, value):
        raise AttributeError("The noise_figure parameter cannot be changed after instantiation of the Fibre")

    @property
    def noise_figure_linear(self):
        """
        [float] the amplifier noise figure in linear gain.
        """
        return dB2linear(self.noise_figure_dB, Gain.POWER)

    @noise_figure_linear.setter
    def noise_figure_linear(self):
        raise AttributeError("The noise_figure parameter cannot be changed after instantiation of the Fibre")

    @property
    def polarisation_mode_dispersion(self) -> float:
        """
        [float] polarisation mode dispersion parameter of this Fibre in ps / (km ^ 0.5)
        """
        return float(self._polarisation_mode_dispersion)

    @polarisation_mode_dispersion.setter
    def polarisation_mode_dispersion(self, value):
        raise AttributeError("The polarisation mode dispersion parameter polarisation_mode_dispersion cannot be changed after instantiation of the Fibre")

    @property
    def realisation_count(self) -> float:
        """
        [int] realisation count of this Fibre
        """
        return int(self._realisation_count)

    @realisation_count.setter
    def realisation_count(self, value):
        raise AttributeError("The realisation count realisation_count cannot be changed after instantiation of the Fibre")

    @property
    def length(self) -> float:
        """
        [float] total length of the fibre in km
        """
        return self.step_path.positions[-1]

    @length.setter
    def length(self, value):
        raise AttributeError("The fibre length cannot be set directly")

    @property
    def photoelasticity(self) -> float:
        """
        [float] fibre photoelasticity constant
        """
        return self._photoelasticity

    @photoelasticity.setter
    def photoelasticity(self, value):
        raise AttributeError("The photoelasticity constant cannot be changed after instantiation of the Fibre")

    @property
    def material(self):
        """
        [RefractiveIndexMaterial] The fibre material, used to determine group velocity from carrier wavelength.
        """
        return self._material

    @material.setter
    def material(self, value):
        raise AttributeError("The fibre material cannot be changed after instantiation")

    @property
    def differential_group_delays(self) -> np.ndarray:
        """
        [float] the differential group delay per step and realisation in ps. Shape [S, R] where S is the number of fibre steps and R the number of realisations.
        """
        return self._differential_group_delays

    @differential_group_delays.setter
    def differential_group_delays(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), f"New differential_group_delays must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (int, float), f"New differential_group_delays array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.step_path.edge_count, self.realisation_count), f"New differential_group_delays array must have shape (self.step_path.edge_count ({self.step_path.edge_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        self._differential_group_delays = value.copy().astype(float)

    @property
    def differential_group_delay(self) -> np.ndarray:
        """
        Accumulated differential group delay in ps, shape [R] where R is the number of fibre realisations
        """
        return self.accumulate_differential_group_delays()

    @differential_group_delay.setter
    def differential_group_delay(self, value):
        raise AttributeError("The accumulated differential_group_delay cannot be set directly; set step differential_group_delays instead")

    @property
    def modulus_model(self) -> ModulusModel:
        """
        The modulus model with which the birefringence is initialised (FIXED or RANDOM)
        """
        return self._modulus_model

    @modulus_model.setter
    def modulus_model(self, value):
        raise AttributeError("The fibre birefringence modulus model cannot be changed after instantiation")
