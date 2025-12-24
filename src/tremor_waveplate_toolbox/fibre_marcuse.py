"""
An optical fibre channel model base class for dual-polarisation transmission, based on Marcuse's method.
Currently it models only polarisation mode dispersion: (perturbed) birefringence and major axes rotations.
"""
from typing import override

import numpy as np

from .fibre import Fibre
from .signal import Signal
from .perturbation import Perturbation
from .utils import rotation_matrix
from .constants import Device

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
    def _init_birefringence(self):
        """
        Generate differential phase shifts, group delays, and major axes orientations, for use in Marcuse's split-step CNLSE.
        """
        # self.section_DGD = np.tile(self.PMD_parameter / np.sqrt(8 * self.correlation_length) * self.section_path.lengths[:, None], (1, self.realisation_count))
        self.differential_phase_shifts = np.tile(2 * np.pi / self.beat_length * self.section_path.lengths[:, None], (1, self.realisation_count))
        self.differential_group_delays = np.tile(self.polarisation_mode_dispersion / np.sqrt(2 * self.correlation_length) * self.section_path.lengths[:, None], (1, self.realisation_count))
        self.major_angles              = np.cumsum(np.sqrt(self.section_path.lengths / (2 * self.correlation_length))[:, None] * np.random.default_rng().normal(size = (self.section_path.edge_count, self.realisation_count)), axis = 0)

    @override
    def _propagate_master(self, signal: Signal, frequency_angular: np.ndarray, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = [], verbose: bool = False) -> Signal:
        """
        Master function both for propagating a signal or building a Jones transfer matrix
        """
        signal, frequency_angular = self._prepare_signal(signal, frequency_angular)
        perturbations             = self._prepare_perturbations(signal, perturbations)
        transmission_start_times  = self._prepare_transmission_start_times(signal, transmission_start_times)
        section_iterable          = self._prepare_section_iterable(signal, perturbations, verbose)

        for section_index, (differential_phase_shift, differential_group_delay, major_angle) in enumerate(section_iterable):
            # Prepare perturbations-related variables
            perturbations_sample_masks, perturbations_sample_indices = self._perturbations_indices(signal, perturbations, transmission_start_times, section_index)

            # Rotate to the (perturbed) local birefringence axes
            major_angle    = self._perturb_major_angle(signal, section_index, major_angle, perturbations, perturbations_sample_masks, perturbations_sample_indices)
            major_rotation = rotation_matrix(major_angle)
            signal         = self._to_major_axes(signal, major_rotation)
            
            # Apply (perturbed/frequency-dependent) differential phase
            birefringence = self._construct_birefringence(differential_phase_shift, differential_group_delay, frequency_angular)
            birefringence = self._perturb_birefringence(signal, section_index, birefringence, perturbations, perturbations_sample_masks, perturbations_sample_indices)
            signal        = self._apply_birefringence(signal, birefringence)

            # Rotate back to the reference birefringence axes
            signal = self._to_major_axes(signal, signal.xp.moveaxis(major_rotation, -1, -2))

        return signal

    def _prepare_signal(self, signal, frequency_angular):
        signal = signal.copy()
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
        if not isinstance(transmission_start_times, (float, int)):
            transmission_start_times = signal.xp.array(transmission_start_times)
            assert len(transmission_start_times.shape) == 1, f"transmission_start_times must have shape [T,], but had shape {transmission_start_times.shape}"
            assert signal.shape[signal.sample_axis - 1] == 1, f"If transmission_start_times has shape [T,] signal must have batch size 1, but this was {signal.shape[signal.sample_axis - 1]}"
        else:
            transmission_start_times = signal.xp.array([transmission_start_times])

        return transmission_start_times

    def _prepare_section_iterable(self, signal, perturbations, verbose):
        differential_phase_shifts = signal.xp.array(self.differential_phase_shifts[:, :, *(None,) * -signal.sample_axis_negative]) # [S, R, 1, 1]/[S, R, 1, 1, 1]
        differential_group_delays = signal.xp.array(self.differential_group_delays[:, :, *(None,) * -signal.sample_axis_negative]) # [S, R, 1, 1]/[S, R, 1, 1, 1]
        major_rotations = signal.xp.array(self.major_angles[:, :, *(None,) * -(1 + signal.sample_axis_negative)]) # [S, R, 1]/[S, R, 1, 1]

        iterable = zip(differential_phase_shifts, differential_group_delays, major_rotations)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(
                iterable,
                total = self.section_path.edge_count,
                desc = f"{"Propagating signal through fibre" if signal.sample_axis_negative == -2 else "Building Jones matrix"} ({'CPU' if signal.device == Device.CPU else 'CUDA'}{', perturbed' if len(perturbations) > 0 else ''})"
            )

        return iterable

    def _perturbations_indices(self, signal, perturbations, transmission_start_times, section_index):
        perturbations_sample_times = transmission_start_times + self.section_path.centre_positions[section_index] / self.group_velocity(signal.carrier_wavelength) # [B,]
        perturbations_sample_masks = []
        perturbations_sample_indices = []
        for perturbation in perturbations:
            perturbations_sample_masks.append((perturbations_sample_times >= 0) & (perturbations_sample_times < perturbation.duration))
            perturbations_sample_indices.append(signal.xp.floor(perturbations_sample_times * perturbation.sample_rate).astype(int))

        return perturbations_sample_masks, perturbations_sample_indices
        
    def _perturb_major_angle(self, signal, section_index, section_major_angle, perturbations, perturbations_sample_masks, perturbations_sample_indices):
        for perturbation, perturbation_sample_mask, perturbation_sample_indices in zip(perturbations, perturbations_sample_masks, perturbations_sample_indices):
            if perturbation.major_angles_adders is None: continue
            section_major_angle = section_major_angle + signal.xp.where( # [R, 1]/[R, 1, 1] + [1, B]/[1, B, 1] = [R, B]/[R, B, 1]
                    perturbation_sample_mask,
                    perturbation.major_angles_adders[section_index, perturbation_sample_indices],
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
        if self.polarisation_mode_dispersion != 0:
            differential_phase_shift = differential_phase_shift + differential_group_delay * frequency_angular * 1e-12 # [R, 1, 1]/[R, 1, 1, 1] + [R, 1, 1]/[R, 1, 1, 1] * [1, 1, F]/[1, 1, F, 1] = [R, 1, F]/[R, 1, F, 1]
        return differential_phase_shift

    def _perturb_birefringence(self, signal, section_index, birefringence, perturbations, perturbations_sample_masks, perturbations_sample_indices):
        for perturbation, perturbation_sample_mask, perturbation_sample_indices in zip(perturbations, perturbations_sample_masks, perturbations_sample_indices):
            if perturbation.birefringence_scalars is None: continue
            birefringence = birefringence * signal.xp.where(
                    perturbation_sample_mask,
                    perturbation.birefringence_scalars[section_index, perturbation_sample_indices],
                    1.0
                )[None, :, *(None,) * -(1 + signal.sample_axis_negative)] # [R, 1, F]/[R, 1, F, 1] * [1, B, 1]/[1, B, 1, 1] = [R, B, F]/[R, B, F, 1]

        for perturbation, perturbation_sample_mask, perturbation_sample_indices in zip(perturbations, perturbations_sample_masks, perturbations_sample_indices):
            if perturbation.birefringence_adders is None: continue
            birefringence = birefringence + signal.xp.where(
                    perturbation_sample_mask,
                    perturbation.birefringence_adders[section_index, perturbation_sample_indices],
                    0.0
                )[None, :, *(None,) * -(1 + signal.sample_axis_negative)] # [R, B, F]/[R, B, F, 1] * [1, B, 1]/[1, B, 1, 1] = [R, B, F]/[R, B, F, 1]

        return birefringence

    def _apply_birefringence(self, signal, birefringence):
        birefringence = signal.xp.exp(-0.5j * birefringence)
        signal.samples_frequency = signal.samples_frequency * signal.xp.stack([birefringence, birefringence.conjugate()], axis = -1) # [R, B, F, 2]/[R, B, F, 2, 2] * [R, B, F, 2]/[R, B, F, 1, 2] = [R, B, F, 2]/[R, B, F, 2, 2]
        return signal

    @override
    def to_dict(self):
        return super().to_dict() | {
                'differential_phase_shifts': self.differential_phase_shifts.tolist(),
                'major_angles':   self.major_angles.tolist()
        }

    @classmethod
    @override
    def from_dict(cls, fibre_dict: dict):
        fibre = super(FibreMarcuse, cls).from_dict(fibre_dict)
        fibre.differential_phase_shifts = np.array(fibre_dict['differential_phase_shifts'])
        fibre.major_angles              = np.array(fibre_dict['major_angles'])
        return fibre

    @override
    def __eq__(self, other) -> bool:
        return super().__eq__(other) and \
            np.all(self._differential_phase_shifts == other._differential_phase_shifts) and \
            np.all(self._major_angles              == other._major_angles)

    @property
    def differential_phase_shifts(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] local differential phase shift between the major polarisation axes per section in radians
        """
        return self._differential_phase_shifts

    @differential_phase_shifts.setter
    def differential_phase_shifts(self, value) -> None:
        assert isinstance(value, np.ndarray), f"New differential_phase_shifts value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New differential_phase_shifts array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.section_path.edge_count, self.realisation_count), f"New differential_phase_shifts array must have shape (self.section_path.edge_count ({self.section_path.edge_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        # assert np.all((value >= -np.pi) & (value < np.pi)), f"New differential_phase_shifts array must have values between -pi and pi"
        self._differential_phase_shifts = value.copy().astype(float)

    @property
    def major_angles(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] orientation of the major axes of birefringence per section in radians
        """
        return self._major_angles

    @major_angles.setter
    def major_angles(self, value) -> None:
        assert isinstance(value, np.ndarray), f"New major_angles value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New major_angles array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.section_path.edge_count, self.realisation_count), f"New major_angles array must have shape (self.section_path.edge_count ({self.section_path.edge_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        # assert np.all((value >= -np.pi) & (value < np.pi)), f"New section_major_angles array must have values between -pi and pi"
        self._major_angles = value.copy().astype(float)