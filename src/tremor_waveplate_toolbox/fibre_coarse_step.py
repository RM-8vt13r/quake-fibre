"""
An optical fibre channel model base class for dual-polarisation transmission, based on the coarse-step method.
Currently it models only PMD effects: (perturbed) differential phase and major birefringence axes rotations.
"""
from typing import override
from tqdm import tqdm
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
            self.polarisation_mode_dispersion * np.sqrt(3 * np.pi * self.section_path.lengths / 8)[:, None], # Czegledi et al. (2016): Polarization-Mode Dispersion Aware Digital Backpropagation, Prola et al. (1997): PMD Emulators and Signal Distortion in 2.48-Gb/s IM-DD Lightwave Systems
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
        Initialise random scramblers between fibre sections that perform a frequency-invariant scrambling of the state of polarisation.
        """
        random_vectors = np.random.default_rng().normal(
           size = (self.section_path.edge_count, self.realisation_count, 4)
        ) # Initialise [cos(theta), a * sin(theta)], Czegledi et al. (2016): Polarization Drift Channel Model for Coherent Fibre-Optic Systems
        random_vectors /= np.linalg.norm(random_vectors, axis = -1)[..., None]

        rotation_angles   = np.arccos(random_vectors[..., 0])
        rotation_axes     = random_vectors[..., 1:] / np.sin(rotation_angles)[..., None]
        self.scramblers   = sp.linalg.expm(-1j * rotation_angles[:, :, None, None] * np.einsum('sra,apq->srpq', rotation_axes, PAULI_VECTOR))

    @override
    def _propagate_master(self, signal: Signal, frequency_angular: np.ndarray, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = []) -> Signal:
        """
        Master function both for propagating a signal or building a Jones transfer matrix
        """
        if perturbations is None: perturbations = ()
        assert len(perturbations) == 0, f"Perturbations not yet implemented in the coarse-step fibre model"
        assert self.chromatic_dispersion == 0, f"Chromatic dispersion not yet implemented in the coarse-step fibre model"
        assert self.nonlinearity == 0, f"Nonlinearity not yet implemented in the coarse-step fibre model"

        if not isinstance(transmission_start_times, (float, int)):
            transmission_start_times = signal.xp.array(transmission_start_times)
            assert len(transmission_start_times.shape) == 1, f"transmission_start_times must have shape [T,], but had shape {transmission_start_times.shape}"
            assert signal.shape[signal.sample_axis - 1] == 1, f"If transmission_start_times has shape [T,] signal must have batch size 1, but this was {signal.shape[signal.sample_axis - 1]}"
        else:
            transmission_start_times = signal.xp.array([transmission_start_times])

        differential_group_delays = signal.xp.array(self.differential_group_delays[:, :, *(None,) * -signal.sample_axis_negative]) # [S, R, 1, 1]/[S, R, 1, 1, 1]
        scramblers = signal.xp.array(self.scramblers[:, :, *(None,) * -(1 + signal.sample_axis_negative)]) # [S, R, 1, 2, 2]/[S, R, 1, 1, 2, 2]

        frequency_angular = frequency_angular[*(None,) * 2, :, *(None,) * -(2 + signal.sample_axis_negative)] # [1, 1, F]/[1, 1, F, 1]

        iterable = zip(differential_group_delays, scramblers)
        if logger.isEnabledFor(logging.DEBUG):
            iterable = tqdm(
                iterable,
                total = self.section_path.edge_count,
                desc = f"{"Propagating signal through fibre" if signal.sample_axis_negative == -2 else "Building Jones matrix"} ({'CPU' if signal.device == Device.CPU else 'CUDA'}{', perturbed' if len(perturbations) > 0 else ''})"
            )

        for differential_group_delay, scrambler in iterable: # [R, 1, 1], [R, 1]
            if self.polarisation_mode_dispersion != 0:
                # Apply local differential group delay
                differential_group_delay = signal.xp.exp(-0.5j * differential_group_delay * frequency_angular * 1e-12) # [R, 1, 1]/[R, 1, 1, 1] * [1, 1, F]/[1, 1, F, 1] = [R, 1, F]/[R, 1, F, 1]
                signal.samples_frequency = signal.samples_frequency * signal.xp.stack([differential_group_delay, differential_group_delay.conjugate()], axis = -1) # [R, B, F, 2]/[R, B, F, 2, 2] * [R, 1, F, 2]/[R, 1, F, 1, 2] = [R, B, F, 2]/[R, B, F, 2, 2]

                # Scramble state of polarisation
                signal.samples_frequency = signal.xp.einsum( # [R, 1, 2, 2]/[R, 1, 1, 2, 2] @ [R, B, F, 2]/[R, B, F, 2, 2] = [R, B, F, 2]/[R, B, F, 2, 2]
                    '...pq,...sq->...sp',
                    scrambler,
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
        assert value.shape == (self.section_path.edge_count, self.realisation_count, 2, 2), f"New scramblers array must have shape (self.section_path.edge_count ({self.section_path.edge_count}), self.realisation_count ({self.realisation_count}), 2, 2), but had shape {value.shape}"
        assert np.allclose(value[..., 0, 0], value[..., 1, 1].conjugate()) and np.allclose(value[..., 1, 0], -value[..., 0, 1].conjugate()), f"New scramblers array must contain unitary matrices, but didn't"
        self._scramblers = value.copy().astype(complex)