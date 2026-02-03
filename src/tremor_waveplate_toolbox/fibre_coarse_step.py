"""
An optical fibre channel model base class for dual-polarisation transmission, based on the coarse-step method.
"""
from typing import override
import logging

import numpy as np
import scipy as sp

from .fibre import Fibre
from .signal import Signal
from .perturbation import Perturbation
from .constants import Device, PAULI_VECTOR

logger = logging.getLogger()

class FibreCoarseStep(Fibre):
    """
    A class representing an optical fibre.
    Currently it models only polarisation-mode dispersion, using the coarse-step method.
    This model applies differential group delay and scrambles the state of polarisation in a completely random and distributed manner.
    Chromatic dispersion, the Kerr effect, attenuation, EDFA noise, and polarisation-dependent loss are neglected, and slow PMD drift are not implemented.
    External perturbations can be modelled as a change in differential phase or major birefringence axes orientations.
    """
    @override
    def _init_birefringence_fixed(self):
        """
        Initialise fixed differential group delays for the coarse-step method.
        """
        self.differential_group_delays = np.tile(
            self.polarisation_mode_dispersion * np.sqrt(3 * np.pi * self.step_path.lengths / 8)[:, None], # Czegledi et al. (2016): Polarization-Mode Dispersion Aware Digital Backpropagation, Prola et al. (1997): PMD Emulators and Signal Distortion in 2.48-Gb/s IM-DD Lightwave Systems
            (1, self.realisation_count)
        )
        self._init_scramblers()

    @override
    def _init_birefringence_random(self):
        """
        Initialise random Gaussian differential group delays for the coarse-step method.
        """
        self._init_birefringence_fixed()
        self.differential_group_delays = np.random.default_rng().normal(
            loc   = self.differential_group_delays,
            scale = self.differential_group_delays / 5
        )
        self._init_scramblers()

    @override
    def _init_scramblers(self):
        """
        Initialise random scramblers between fibre steps that perform a frequency-invariant scrambling of the state of polarisation.
        """
        random_vectors = np.random.default_rng().normal(
           size = (self.step_path.edge_count, self.realisation_count, 4)
        ) # Initialise [cos(theta), a * sin(theta)], Czegledi et al. (2016): Polarization Drift Channel Model for Coherent Fibre-Optic Systems
        random_vectors /= np.linalg.norm(random_vectors, axis = -1)[..., None]

        rotation_angles   = np.arccos(random_vectors[..., 0])
        rotation_axes     = random_vectors[..., 1:] / np.sin(rotation_angles)[..., None]
        self.scramblers   = sp.linalg.expm(-1j * rotation_angles[:, :, None, None] * np.einsum('sra,apq->srpq', rotation_axes, PAULI_VECTOR))

    @override
    def propagate(self, signal: Signal, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = [], step_start: int = None, step_stop: int = None) -> Signal:
        """
        Master function both for propagating a signal or building a Jones transfer matrix
        """
        assert len(perturbations) == 0, f"Perturbations not yet implemented in the coarse-step fibre model"
        assert self.chromatic_dispersion == 0, f"Chromatic dispersion not yet implemented in the coarse-step fibre model"
        assert self.nonlinearity == 0, f"Nonlinearity not yet implemented in the coarse-step fibre model"
        # assert np.isinf(self.attenuation), f"Attenuation not yet implemented in the coarse-step fibre model"

        return super().propagate(signal, transmission_start_times, perturbations, step_start, step_stop)

    @override
    def _prepare_steps_iterable_arrays(self, signal, step_start, step_stop):
        step_lengths, differential_group_delays = super()._prepare_steps_iterable_arrays(signal, step_start, step_stop)
        scramblers = signal.invite_array(self.scramblers[step_start:step_stop, :, None]) # [S, R, 1, 2, 2]
        return step_lengths, differential_group_delays, scramblers

    def _perturb_birefringence_quantities(self, signal, step_index, perturbations, perturbation_sample_masks, perturbation_sample_indices, *birefringence_quantities):
        pass

    @override
    def _prepare_birefringence(self, signal, differential_group_delay, scrambler):
        return signal

    @override
    def _apply_half_birefringence(self, linear_exponent, signal, differential_group_delay, scrambler):
        birefringence_exponent = 0.25j * differential_group_delay * signal.frequency_angular * 1e-12 # [R, B, 1] * [F] = [R, B, F]. 0.25 because 1) we have two polarisations and 2) we apply only half the birefringence
        linear_exponent[:, :, :, 0] += birefringence_exponent
        linear_exponent[:, :, :, 1] -= birefringence_exponent
        return linear_exponent

    @override
    def _finalise_birefringence(self, signal, differential_group_delay, scrambler):
        signal = self._scramble(signal, scrambler)
        return signal

    def _scramble(self, signal, scrambler):
        signal.samples_frequency = signal.xp.einsum( # [R, 1, 2, 2] @ [R, B, F, 2] = [R, B, F, 2]
                'rbpq,rbsq->rbsp',
                scrambler, # rotation_matrix(-major_angle) = rotation_matrix(major_angle).transpose
                signal.samples_frequency,
                optimize = True
            )
        return signal

    @override
    def to_dict(self):
        return super().to_dict() | {
                'scramblers': self.scramblers.tolist()
            }

    @classmethod
    @override
    def from_dict(cls, fibre_dict: dict):
        fibre = super(FibreCoarseStep, cls).from_dict(fibre_dict)
        fibre.scramblers = np.array(fibre_dict['scramblers'])
        return fibre

    @override
    def __eq__(self, other) -> bool:
        return super().__eq__(other) and \
            np.all(self.scramblers == other.scramblers)

    @property
    def scramblers(self) -> np.ndarray:
        """
        [np.ndarray], dtype [complex] unitary matrices that scramble the state of polarisation
        """
        return self._scramblers

    @scramblers.setter
    def scramblers(self, value) -> None:
        assert isinstance(value, np.ndarray), f"New scramblers value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int, complex), f"New scramblers array must contain values of type complex, but contained {value.dtype}"
        assert value.shape == (self.step_path.edge_count, self.realisation_count, 2, 2), f"New scramblers array must have shape (self.step_path.edge_count ({self.step_path.edge_count}), self.realisation_count ({self.realisation_count}), 2, 2), but had shape {value.shape}"
        assert np.allclose(value[..., 0, 0], value[..., 1, 1].conjugate()) and np.allclose(value[..., 1, 0], -value[..., 0, 1].conjugate()), f"New scramblers array must contain unitary matrices, but didn't"
        self._scramblers = value.copy().astype(complex)