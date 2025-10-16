"""
An optical fibre channel model base class for dual-polarisation transmission, based on Marcuse's method.
Currently it models only PMD effects: (earthquake-dependent) differential group delay and state of polarisation scramblers.
"""
from configparser import ConfigParser
from typing import override

import numpy as np

from .fibre import Fibre
from .signal import Signal
from .utils import rotation_matrix, phase_matrix

class FibreMarcuse(Fibre):
    """
    A class representing an optical fibre.
    Currently it models only polarisation-mode dispersion, using Marcuse's method.
    This method models the slow rotation of the major birefringence axes over a section length much shorter than the correlation length.
    Each section, the major birefringence axes are rotated to the local preference, differential phase is added, and the SOP is rotated back to the global reference frame.
    Chromatic dispersion, the Kerr effect, attenuation, EDFA noise, and polarisation-dependent loss are neglected, and slow PMD drift are not implemented.
    Earthquake strain can be modelled as a change in differential group delay.
    """
    @override
    def _init_DGD(self):
        """
        Marcuse's method assumes differential phase shifts to be uniformly distributed over the fibre length.
        """
        self.section_differential_phase = np.tile(2 * np.pi / self.beat_length * self.section_lengths[:, None], (1, self.realisation_count))
        self.section_DGD = np.tile(2 * self.PMD_parameter / np.sqrt(8 * self.correlation_length) * self.section_lengths[:, None], (1, self.realisation_count))

    @override
    def _init_PSP(self):
        """
        In Marcuse's method, PSPs of subsequent sections are strongly correlated, and DGD is distributed uniformly along the fibre.
        """
        self._section_birefringences = 0
        self.section_birefringences  = np.full(
            fill_value = np.pi / self.beat_length,
            shape = (len(self.section_lengths), self.realisation_count)
        )
        self._section_major_angles = 0
        self.section_major_angles  = np.cumsum(np.sqrt(self.section_lengths / (2 * self.correlation_length))[:, None] * np.random.default_rng().normal(size = (self.section_count, self.realisation_count)), axis = 0)
        self._section_PSP          = None

    @override
    def propagate(self, signal: Signal, strain: Signal = None, verbose: bool = False) -> Signal:
        """
        This model rotates to the correlated local major birefringence axes, applies a uniform differential phase- and group delay, and rotates back to the reference frame in every section.
        """
        assert strain is None, f"earthquake functionality not implemented yet"

        if signal.device == Device.CUDA:
            xp = cp
            signal.to_device(Device.CUDA) # Ensure that the signal resides in the currently active cupy GPU
        else:
            xp = np

        section_DGDs = xp.array(self.section_DGD)
        section_major_angles = xp.array(self.section_major_angles)
        section_birefringences = xp.array(self.section_birefringences)

        signal = signal.copy()
        signal.samples_frequency = xp.tile(signal.samples_frequency, (self.realisation_count, 1, 1, 1))
        frequency_angular = signal.frequency_angular[None, None]

        iterable = zip(section_DGDs, section_major_angles, section_birefringences)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(
                iterable,
                total = self.section_count,
                desc = "Propagating signal through fibre"
            )

        for section_DGD, section_major_angle, section_birefringence in iterable:
            # Rotate to local birefringence axes
            signal.samples_frequency = xp.einsum(
                'rpq,rbsq->rbsp',
                rotation_matrix(section_major_angle),
                signal.samples_frequency
            )

            # Apply differential phase
            differential_phase = xp.exp(-0.5j * (section_birefringence[:, None, None] + section_DGD[:, None, None] * frequency_angular * 1e-12))
            signal.samples_frequency[..., 0] *= differential_phase
            signal.samples_frequency[..., 1] *= differential_phase.conjugate()

            # Rotate back
            signal.samples_frequency = xp.einsum(
                'rpq,rbsq->rbsp',
                rotation_matrix(-section_major_angle),
                signal.samples_frequency
            )

        return signal

    @override
    def Jones(self, frequency_angular: (np.ndarray, cp.ndarray), verbose: bool = False) -> np.ndarray:
        assert len(frequency_angular.shape) == 1, f"frequency_angular must have shape [F,], but had shape {frequency_angular.shape}"
        
        if signal.device == Device.CUDA:
            xp = cp
            signal.to_device(Device.CUDA) # Ensure that the signal resides in the currently active cupy GPU
        else:
            xp = np

        section_DGDs = xp.array(self.section_DGD)
        section_major_angles = xp.array(self.section_major_angles)
        section_birefringences = xp.array(self.section_birefringences)

        iterable = zip(section_DGD, section_major_angles, section_birefringences)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(
                iterable,
                total = self.section_count,
                desc = "Building Jones matrix"
            )
        
        Jones_matrix = cp.eye(2, dtype = complex)[None, None] # [1, 1, 2, 2]
        for section_DGD, section_major_angle, section_birefringence in iterable:
            # Rotate to local birefringence axes
            Jones_matrix = rotation_matrix(section_major_angle)[:, None] @ Jones_matrix

            # Apply differential phase
            differential_phase = cp.exp(-0.5j * (section_birefringence[:, None] + section_DGD[:, None] * frequency_angular[None, :] * 1e-12))[:, :, None]
            Jones_matrix = Jones_matrix * cp.stack([differential_phase, differential_phase.conjugate()], axis = 2)

            # Rotate back
            Jones_matrix = rotation_matrix(-section_major_angle)[:, None] @ Jones_matrix

        return Jones_matrix