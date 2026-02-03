"""
An optical fibre channel model base class for dual-polarisation transmission, based on the split-step method for the coupled nonlinear Schrödinger equation.
"""
from typing import override
import sys
import logging

import numpy as np
try:
    import cupy as cp
except:
    pass

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

class FibreCNLSE(Fibre):
    """
    A class representing an optical fibre.
    Currently it models only polarisation-mode dispersion, using the split-step coupled nonlinear Schrödinger equation from Marcuse et al. (1997).
    This method models the rotation of the major birefringence axes over a step length much shorter than the correlation length.
    Each step, the major birefringence axes are rotated to the local preference, differential phase is added, and the SOP is rotated back to the global reference frame.
    External perturbations can be modelled as a change in birefringence or major birefringence axes orientations.
    """
    @override
    def _init_birefringence_fixed(self):
        """
        Initialise fixed birefringences and a random walk for the major birefringence angles for the split-step method, using the fixed modulus model.
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
        Initialise random birefringences and major birefringence angles for the split-step method using the random modulus model (a Langevin process).
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
    def _prepare_steps_iterable_arrays(self, signal, step_start, step_stop):
        step_lengths, differential_group_delays = super()._prepare_steps_iterable_arrays(signal, step_start, step_stop)
        differential_phase_shifts = signal.invite_array(self.differential_phase_shifts[step_start:step_stop, :, None, None]) # [K, R, 1, 1]
        major_rotation_matrices   = signal.invite_array(rotation_matrix(self.major_angles[step_start:step_stop, :, None])) # [K, R, 1, 2, 2]
        return step_lengths, differential_group_delays, differential_phase_shifts, major_rotation_matrices

    @override
    def _perturb_birefringence_quantities(self, signal, step_index, perturbations, perturbations_sample_masks, perturbations_sample_indices, differential_group_delay, differential_phase_shift, major_rotation_matrix):
        for perturbation, perturbation_sample_mask, perturbation_sample_indices in zip(perturbations, perturbations_sample_masks, perturbations_sample_indices):
            if perturbation.strains is not None:
                birefringence_components = signal.xp.stack((differential_group_delay, differential_phase_shift), axis = 0) # [2, R, 1, 1]
                differential_group_delay, differential_phase_shift = birefringence_components * signal.xp.where( # [2, R, 1, 1] * [1, 1, B, 1] = [2, R, B, 1]
                        perturbation_sample_mask,
                        1.0 + self.photoelasticity * perturbation.strains[step_index, perturbation_sample_indices],
                        1.0
                    )[None, None, :, None]

            if perturbation.twists is not None:
                perturbing_rotation_matrix = signal.xp.where(
                        perturbation_sample_mask,
                        rotation_matrix(perturbation.twists[step_index, perturbation_sample_indices]),
                        0.0
                    )[None]

                major_rotation_matrix = signal.xp.einsum( # [R, 1, 2, 2] @ [1, B, 2, 2] = [R, B, 2, 2]
                    "rbpq,rbsq->rbsp",
                    major_rotation_matrix,
                    perturbing_rotation_matrix,
                    optimize = True
                )

        return differential_group_delay, differential_phase_shift, major_rotation_matrix

    @override
    def _prepare_birefringence(self, signal, differential_group_delay, differential_phase_shift, major_rotation_matrix):
        signal = self._rotate_major_axes(signal, signal.xp.moveaxis(major_rotation_matrix, -2, -1))
        return signal

    @override
    def _apply_half_birefringence(self, linear_exponent, signal, differential_group_delay, differential_phase_shift, major_rotation_matrix):
        birefringence_exponent = 0.25j * (differential_phase_shift + differential_group_delay * signal.frequency_angular * 1e-12) # [R, B, 1] * [F] = [R, B, F]. 0.25 because 1) we have two polarisations and 2) we apply only half the birefringence
        linear_exponent[:, :, :, 0] += birefringence_exponent
        linear_exponent[:, :, :, 1] -= birefringence_exponent
        return linear_exponent

    @override
    def _finalise_birefringence(self, signal, differential_group_delay, differential_phase_shift, major_rotation_matrix):
        signal = self._rotate_major_axes(signal, major_rotation_matrix)
        return signal

    def _rotate_major_axes(self, signal, major_rotation_matrix):
        signal.samples_frequency = signal.xp.einsum( # [R, 1, 2, 2] @ [R, B, S, 2] = [R, B, S, 2]
                'rbpq,rbsq->rbsp',
                major_rotation_matrix, # Multiply with the transposed rotation matrix, such that we effectively rotate the axes and not the signal
                signal.samples_frequency,
                optimize = True
            )
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
        fibre = super(FibreCNLSE, cls).from_dict(fibre_dict)
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
        [np.ndarray], dtype [float] local differential phase shift between the major polarisation axes per step in radians. Shape [S, R] where S is the number of fibre steps and R the number of realisations.
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
        [np.ndarray], dtype [float] orientation of the major axes of birefringence per step in radians. Shape [S, R] where S is the number of fibre steps and R the number of realisations.
        """
        return self._major_angles

    @major_angles.setter
    def major_angles(self, value) -> None:
        assert isinstance(value, np.ndarray), f"New major_angles value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New major_angles array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.step_path.edge_count, self.realisation_count), f"New major_angles array must have shape (self.step_path.edge_count ({self.step_path.edge_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        # assert np.all((value >= -np.pi) & (value < np.pi)), f"New step_major_angles array must have values between -pi and pi"
        self._major_angles = value.copy().astype(float)