"""
An optical fibre channel model base class for dual-polarisation transmission, based on Marcuse's method.
Currently it models only polarisation mode dispersion: (perturbed) birefringence and major axes rotations.
"""
from typing import override
from tqdm import tqdm
import logging

import numpy as np

from .fibre import Fibre
from .signal import Signal
from .perturbation import Perturbation
from .utils import rotation_matrix
from .constants import Device, PAULI_3
try:
    from .constants import PAULI_3_CUDA
except:
    pass

logger = logging.getLogger()

class FibreMarcuse(Fibre):
    """
    A class representing an optical fibre.
    Currently it models only polarisation-mode dispersion, using Marcuse's method.
    This method models the slow rotation of the major birefringence axes over a section length much shorter than the correlation length.
    Each section, the major birefringence axes are rotated to the local preference, differential phase is added, and the SOP is rotated back to the global reference frame.
    Chromatic dispersion, the Kerr effect, attenuation, EDFA noise, and polarisation-dependent loss are neglected, and slow polarisation mode dispersion drift are not implemented.
    External perturbations can be modelled as a change in differential phase or major birefringence axes orientations.
    """
    @override
    def _init_birefringence_fixed(self):
        """
        Initialise fixed birefringences and a random walk for the major birefringence angles for Marcuse's split-step method, using the fixed modulus model.
        """
        self.differential_phase_shifts = np.tile(2 * np.pi / self.beat_length * self.step_path.lengths[:, None], (1, self.realisation_count))
        self.differential_group_delays = np.tile(self.polarisation_mode_dispersion / np.sqrt(2 * self.correlation_length) * self.step_path.lengths[:, None], (1, self.realisation_count))

        first_major_angle = 0
        noise = np.random.default_rng().normal(size = (self.step_path.edge_count, self.realisation_count))
        major_angle_steps = np.sqrt(self.step_path.lengths / (2 * self.correlation_length))[:, None] * noise
        self.major_angles = np.cumsum(major_angle_steps, axis = 0)
    
    @override
    def _init_birefringence_random(self):
        """
        Initialise random birefringences and major birefringence angles for Marcuse's split-step method using the random modulus model (a Langevin process).
        """
        # Initialise differential phase shift and major birefringence angles jointly
        # first_phase_x = 2 * np.pi / self.beat_length * self.step_path.lengths[0]
        noise_x = np.random.default_rng().normal(size = (self.step_path.edge_count, self.realisation_count))
        phase_steps_x = np.sqrt(2 * np.pi ** 2 * (np.exp(2 * self.step_path.lengths / self.correlation_length) - 1) * self.step_path.lengths ** 2 / self.beat_length ** 2)[:, None] * noise_x
        # phase_steps_x = np.concatenate([np.full(fill_value = first_phase_x, shape = (1, self.realisation_count)), phase_steps_x], axis = 0)
        phases_x = np.cumsum(phase_steps_x, axis = 0)
        phases_x[1:] = phases_x[1:] / np.exp(self.step_path.lengths[1:] / self.correlation_length)[:, None]

        # first_phase_y = 0
        noise_y = np.random.default_rng().normal(size = (self.step_path.edge_count, self.realisation_count))
        phase_steps_y = np.sqrt(2 * np.pi ** 2 * (np.exp(2 * self.step_path.lengths / self.correlation_length) - 1) * self.step_path.lengths ** 2 / self.beat_length ** 2)[:, None] * noise_y
        # phase_steps_y = np.concatenate([np.full(fill_value = first_phase_y, shape = (1, self.realisation_count)), phase_steps_y], axis = 0)
        phases_y = np.cumsum(phase_steps_y, axis = 0)
        phases_y[1:] = phases_y[1:] / np.exp(self.step_path.lengths[1:] / self.correlation_length)[:, None]

        self.differential_phase_shifts = np.sqrt(phases_x ** 2 + phases_y ** 2)
        self.major_angles = np.arctan2(phases_y, phases_x) / 2

        # Initialise differential group delay
        noise = np.random.default_rng().normal(size = (self.step_path.edge_count, self.realisation_count))
        self.differential_group_delays = np.sqrt(self.polarisation_mode_dispersion ** 2 * self.step_path.lengths ** 3 / (2 * self.correlation_length ** 2 * (np.exp(-self.step_path.lengths / self.correlation_length) + self.step_path.lengths / self.correlation_length - 1)))[:, None] * noise

    @override
    def _propagate_master(self, signal: Signal, frequency_angular: np.ndarray, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = [], step_start: int = None, step_stop: int = None) -> Signal:
        """
        Master function both for propagating a signal or building a Jones transfer matrix. See documentation in fibre.py.
        """
        super()._propagate_master(signal, frequency_angular, transmission_start_times, perturbations, step_start, step_stop)

        signal, frequency_angular = self._prepare_signal(signal, frequency_angular)
        perturbations             = self._prepare_perturbations(signal, perturbations)
        transmission_start_times  = self._prepare_transmission_start_times(signal, transmission_start_times)
        section_iterable          = self._prepare_section_iterable(signal, perturbations, step_start, step_stop)

        for section_index, (section_length, differential_phase_shift, differential_group_delay, major_angle) in enumerate(section_iterable):
            if self.chromatic_dispersion != 0.:
                # Apply half of chromatic dispersion
                signal = self._apply_chromatic_dispersion(signal, section_length / 2)

            if self.polarisation_mode_dispersion != 0.:
                # Prepare perturbations-related variables
                perturbations_sample_masks, perturbations_sample_indices = self._perturbations_indices(signal, perturbations, transmission_start_times, section_index)

                # Rotate to the (perturbed) local birefringence axes
                major_angle    = self._perturb_major_angle(signal, section_index, major_angle, perturbations, perturbations_sample_masks, perturbations_sample_indices)
                major_rotation = rotation_matrix(-major_angle)
                signal         = self._to_major_axes(signal, major_rotation)
                
                # Apply half of the (perturbed/frequency-dependent) differential phase
                birefringence = self._construct_birefringence(differential_phase_shift, differential_group_delay, frequency_angular)
                birefringence = self._perturb_birefringence(signal, section_index, birefringence, perturbations, perturbations_sample_masks, perturbations_sample_indices)
                signal        = self._apply_birefringence(signal, birefringence / 2)

            if self.nonlinearity != 0.:
                # Apply nonlinearity
                signal = self._apply_nonlinearity(signal, section_length)

            if self.attenuation > -np.inf:
                # Apply attenuation
                signal = self._apply_attenuation(signal, section_length)

            if self.chromatic_dispersion != 0.:
                # Apply the second half of chromatic dispersion
                signal = self._apply_chromatic_dispersion(signal, section_length / 2)

            if self.polarisation_mode_dispersion != 0.:
                # Apply the second half of polarisation mode dispersion
                signal = self._apply_birefringence(signal, birefringence / 2)

                # Rotate back to the reference birefringence axes
                signal = self._to_major_axes(signal, signal.xp.moveaxis(major_rotation, -1, -2))

        return signal

    def _prepare_signal(self, signal, frequency_angular):
        if signal.device == Device.CUDA: signal.to_device(Device.CUDA) # Ensure that the signal resides in the currently active GPU
        frequency_angular = frequency_angular[*(None,) * 2, :, *(None,) * -(2 + signal.sample_axis_negative)] # [1, 1, F]/[1, 1, F, 1]
        return signal, frequency_angular

    def _prepare_perturbations(self, signal, perturbations):
        if not isinstance(perturbations, (list, tuple)):
            perturbations = (perturbations,)

        for perturbation in perturbations:
            assert isinstance(perturbation, Perturbation), f"All perturbations must have type Perturbation, but at least one had type {type(perturbation)}"
            perturbation.to_device(signal.device)

        return perturbations
        
    def _prepare_transmission_start_times(self, signal, transmission_start_times):
        if not isinstance(transmission_start_times, (int, np.integer, float, np.floating)):
            transmission_start_times = signal.xp.array(transmission_start_times)
            assert len(transmission_start_times.shape) == 1, f"transmission_start_times must have shape [T,], but had shape {transmission_start_times.shape}"
            assert signal.shape[signal.sample_axis - 1] == 1, f"If transmission_start_times has shape [T,] signal must have batch size 1, but this was {signal.shape[signal.sample_axis - 1]}"
        else:
            transmission_start_times = signal.xp.array([transmission_start_times])

        return transmission_start_times

    def _prepare_section_iterable(self, signal, perturbations, step_start, step_stop):
        differential_phase_shifts = signal.xp.array(self.differential_phase_shifts[step_start:step_stop, :, *(None,) * -signal.sample_axis_negative]) # [S, R, 1, 1]/[S, R, 1, 1, 1]
        differential_group_delays = signal.xp.array(self.differential_group_delays[step_start:step_stop, :, *(None,) * -signal.sample_axis_negative]) # [S, R, 1, 1]/[S, R, 1, 1, 1]
        major_rotations = signal.xp.array(self.major_angles[step_start:step_stop, :, *(None,) * -(1 + signal.sample_axis_negative)]) # [S, R, 1]/[S, R, 1, 1]

        iterable = zip(self.step_path.lengths[step_start:step_stop], differential_phase_shifts, differential_group_delays, major_rotations)

        desc_string = "Propagating signal " if signal.sample_axis_negative == -2 else "Building Jones matrix "
        if step_start is not None: desc_string += f"from fibre step {step_start + 1} "
        if step_stop is not None: desc_string += f"until step {step_stop} of {self.step_path.edge_count} "
        desc_string += "("
        desc_string += "CPU" if signal.device == Device.CPU else "CUDA"
        if len(perturbations): desc_string += ", perturbed"
        desc_string += ")"
        if logger.isEnabledFor(logging.INFO):
            iterable = tqdm(
                iterable,
                total = (self.step_path.edge_count if step_stop is None else step_stop) - (0 if step_start is None else step_start),
                desc = desc_string
            )

        return iterable

    def _apply_chromatic_dispersion(self, signal, section_length):
        signal.samples_frequency = signal.samples_frequency * signal.xp.exp(1j * 0.5 * self.chromatic_dispersion * signal.frequency_angular ** 2 * section_length * 1e-24)[None, None, :, None]
        return signal

    def _apply_nonlinearity(self, signal, section_length):
        """
        Nonlinearity is applied only when propagating a signal (not when building a Jones matrix). Therefore, signal always has shape [R, B, S, P] here with realisations R, batch size B, time/frequency axis S and polarisations P = 2
        """
        # signal.samples_time = signal.samples_time * signal.xp.exp(1j * 8 / 9 * self.nonlinearity * section_length * np.linalg.norm(signal.samples_time, axis = -1)[:, :, :, None] ** 2) # ??? Wrong for now, look at Marcuse's paper
        signal_power   = signal.xp.linalg.norm(signal.samples_time, axis = -1)[:, :, :, None, None] ** 2 # [R, B, S, 1, 1]

        signal_rotator = -1/3 * signal.xp.einsum(
            'rbsp,pq,rbsq->rbs',
            signal.samples_time,
            PAULI_3 if signal.device == Device.CPU else PAULI_3_CUDA,
            signal.samples_time,
            optimize = True
        )[:, :, :, None, None] * (PAULI_3 if signal.device == Device.CPU else PAULI_3_CUDA) # [R, B, S, 2, 2]
        
        nonlinearity_operator = signal.xp.exp(1j * self.nonlinearity * (signal_power + signal_rotator) * section_length) # Eq. 2 in Marcuse et al. (1997), Eq. 5 in Menyuk et al. (1987). ??? halve kerr parameter as in Menyuk et al?
        nonlinearity_operator = signal.xp.moveaxis(nonlinearity_operator, (-1, -2)) # Transpose operator for efficient einsum
        
        signal.samples_time = signal.xp.einsum(
            'rbsqp,rbsq->rbsp',
            nonlinearity_operator,
            signal.samples_time,
            optimize = True
        )

        return signal

    def _apply_attenuation(self, signal, section_length):
        """
        Attenuation is applied only when propagating a signal (not when building a Jones matrix). Therefore, signal always has shape [R, B, S, P] here with realisations R, batch size B, time/frequency axis S and polarisations P = 2
        """
        attenuation_linear = 10 ** (self.attenuation * section_length / 20)
        signal.samples = signal.xp.exp(-self.attenuation_linear / 2 * section_length) * signal.samples
        return signal

    def _perturbations_indices(self, signal, perturbations, transmission_start_times, section_index):
        perturbations_sample_times = transmission_start_times + self.step_path.centre_positions[section_index] / self.group_velocity(signal.carrier_wavelength) # [B,]
        perturbations_sample_masks = []
        perturbations_sample_indices = []
        for perturbation in perturbations:
            perturbations_sample_masks.append((perturbations_sample_times >= perturbation.start_time) & (perturbations_sample_times < perturbation.start_time + perturbation.duration))
            perturbations_sample_indices.append(signal.xp.floor((perturbations_sample_times - perturbation.start_time) * perturbation.sample_rate).astype(int))

        return perturbations_sample_masks, perturbations_sample_indices
        
    def _perturb_major_angle(self, signal, section_index, section_major_angle, perturbations, perturbations_sample_masks, perturbations_sample_indices):
        for perturbation, perturbation_sample_mask, perturbation_sample_indices in zip(perturbations, perturbations_sample_masks, perturbations_sample_indices):
            if perturbation.twists is None: continue
            section_major_angle = section_major_angle + signal.xp.where( # [R, 1]/[R, 1, 1] + [1, B]/[1, B, 1] = [R, B]/[R, B, 1]
                    perturbation_sample_mask,
                    perturbation.twists[section_index, perturbation_sample_indices],
                    0.0
                )[None, :, *(None,) * -(2 + signal.sample_axis_negative)]

        return section_major_angle

    def _to_major_axes(self, signal, section_major_rotation):
        signal.samples_frequency = signal.xp.einsum( # [R, B, 2, 2]/[R, B, 1, 2, 2] @ [R, B, F, 2]/[R, B, F, 2, 2] = [R, B, F, 2]/[R, B, F, 2, 2]
            '...pq,...sq->...sp',
            section_major_rotation,
            signal.samples_frequency,
            optimize = True
        )

        return signal

    def _construct_birefringence(self, differential_phase_shift, differential_group_delay, frequency_angular):
        differential_phase_shift = differential_phase_shift + differential_group_delay * frequency_angular * 1e-12 # [R, 1, 1]/[R, 1, 1, 1] + [R, 1, 1]/[R, 1, 1, 1] * [1, 1, F]/[1, 1, F, 1] = [R, 1, F]/[R, 1, F, 1]
        return differential_phase_shift

    def _perturb_birefringence(self, signal, section_index, birefringence, perturbations, perturbations_sample_masks, perturbations_sample_indices):
        for perturbation, perturbation_sample_mask, perturbation_sample_indices in zip(perturbations, perturbations_sample_masks, perturbations_sample_indices):
            if perturbation.strains is None: continue
            birefringence = birefringence * signal.xp.where(
                    perturbation_sample_mask,
                    1.0 + self.photoelasticity * perturbation.strains[section_index, perturbation_sample_indices],
                    1.0
                )[None, :, *(None,) * -(1 + signal.sample_axis_negative)] # [R, 1, F]/[R, 1, F, 1] * [1, B, 1]/[1, B, 1, 1] = [R, B, F]/[R, B, F, 1]

        # for perturbation, perturbation_sample_mask, perturbation_sample_indices in zip(perturbations, perturbations_sample_masks, perturbations_sample_indices):
        #     if perturbation.differential_phase_shifts is None: continue
        #     birefringence = birefringence + signal.xp.where(
        #             perturbation_sample_mask,
        #             perturbation.differential_phase_shifts[section_index, perturbation_sample_indices],
        #             0.0
        #         )[None, :, *(None,) * -(1 + signal.sample_axis_negative)] # [R, B, F]/[R, B, F, 1] * [1, B, 1]/[1, B, 1, 1] = [R, B, F]/[R, B, F, 1]

        return birefringence

    def _apply_birefringence(self, signal, birefringence):
        birefringence = signal.xp.exp(0.5j * birefringence)
        signal.samples_frequency = signal.samples_frequency * signal.xp.stack([birefringence, birefringence.conjugate()], axis = -1) # [R, B, F, 2]/[R, B, F, 2, 2] * [R, B, F, 2]/[R, B, F, 1, 2] = [R, B, F, 2]/[R, B, F, 2, 2]
        return signal

    @override
    def to_dict(self):
        return super().to_dict() | {
                'differential_phase_shifts': self.differential_phase_shifts.tolist(),
                'major_angles':              self.major_angles.tolist()
            }

    @classmethod
    @override
    def from_dict(cls, fibre_dict: dict):
        fibre = super(FibreMarcuse, cls).from_dict(fibre_dict)
        fibre.differential_phase_shifts = np.array(fibre_dict['differential_phase_shifts'])
        fibre.major_angles              = np.array(fibre_dict['major_angles'])
        return fibre

    # @override
    # def _build_piece_dict(self, base_piece_dict: dict, piece_step_start: int, steps_per_piece: int, piece_step_path: Path) -> dict:
    #     """
    #     Part of the pieces() method.
    #     """
    #     base_piece_dict = super()._build_piece_dict(base_piece_dict, piece_step_start, steps_per_piece, piece_step_path)
    #     base_piece_dict['differential_phase_shifts'] = self.differential_phase_shifts[piece_step_start:piece_step_start + steps_per_piece]
    #     base_piece_dict['major_angles'] = self.major_angles[piece_step_start:piece_step_start + steps_per_piece]
    #     return base_piece_dict

    @override
    def __eq__(self, other) -> bool:
        return super().__eq__(other) and \
            np.all(self._differential_phase_shifts == other._differential_phase_shifts) and \
            np.all(self._major_angles              == other._major_angles)

    @property
    def differential_phase_shifts(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] local differential phase shift between the major polarisation axes per section in radians. Shape [S, R] where S is the number of fibre sections and R the number of realisations.
        """
        return self._differential_phase_shifts

    @differential_phase_shifts.setter
    def differential_phase_shifts(self, value) -> None:
        assert isinstance(value, np.ndarray), f"New differential_phase_shifts value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New differential_phase_shifts array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.step_path.edge_count, self.realisation_count), f"New differential_phase_shifts array must have shape (self.step_path.edge_count ({self.step_path.edge_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        # assert np.all((value >= -np.pi) & (value < np.pi)), f"New differential_phase_shifts array must have values between -pi and pi"
        self._differential_phase_shifts = value.copy().astype(float)

    @property
    def major_angles(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] orientation of the major axes of birefringence per section in radians. Shape [S, R] where S is the number of fibre sections and R the number of realisations.
        """
        return self._major_angles

    @major_angles.setter
    def major_angles(self, value) -> None:
        assert isinstance(value, np.ndarray), f"New major_angles value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New major_angles array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.step_path.edge_count, self.realisation_count), f"New major_angles array must have shape (self.step_path.edge_count ({self.step_path.edge_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        # assert np.all((value >= -np.pi) & (value < np.pi)), f"New section_major_angles array must have values between -pi and pi"
        self._major_angles = value.copy().astype(float)