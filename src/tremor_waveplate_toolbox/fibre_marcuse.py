"""
An optical fibre channel model base class for dual-polarisation transmission, based on Marcuse's method.
Currently it models only PMD effects: (earthquake-dependent) differential group delay and state of polarisation scramblers.
"""
from configparser import ConfigParser
from typing import override
import sys

import time

import numpy as np
try:
    import cupy as cp
except:
    pass

from .fibre import Fibre
from .signal import Signal
from .utils import rotation_matrix, phase_matrix
from .constants import Device, PAULI_1, PAULI_2

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
    def propagate(self, signal: Signal, strain: Signal = None, transmission_time: float = 0, verbose: bool = False) -> Signal:
        """
        This model rotates to the correlated local major birefringence axes, applies a uniform differential phase- and group delay, and rotates back to the reference frame in every section.
        """
        if signal.device == Device.CUDA: signal.to_device(Device.CUDA) # Ensure that the signal resides in the currently active cupy GPU
        if strain is not None: strain.to_device(signal.device)

        section_DGDs = signal.xp.array(self.section_DGD[:, :, None, None])
        section_major_rotations = signal.xp.array(rotation_matrix(self.section_major_angles[:, :, None]))
        section_birefringences = signal.xp.array(self.section_birefringences[:, :, None, None])

        signal = signal.copy()
        signal.samples_frequency = signal.xp.tile(signal.samples_frequency, (self.realisation_count, 1, 1, 1))
        frequency_angular = signal.frequency_angular[None, None]

        iterable = zip(section_DGDs, section_major_rotations, section_birefringences)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(
                iterable,
                total = self.section_count,
                desc = f"Propagating signal through fibre ({'CPU' if signal.device == Device.CPU else 'CUDA'})"
            )

        for section_DGD, section_major_rotation, section_birefringence in iterable:
            # Rotate to local birefringence axes
            signal.samples_frequency = signal.xp.einsum(
                'rbpq,rbsq->rbsp',
                section_major_rotation,
                signal.samples_frequency,
                optimize = True
            )
            
            # Apply differential phase
            differential_phase = signal.xp.exp(-0.5j * (section_birefringence + section_DGD * frequency_angular * 1e-12))
            signal.samples_frequency *= signal.xp.stack([differential_phase, differential_phase.conjugate()], axis = 3) # [R, B, S, 2] * [R, 1, S]

            # Apply strain
            if strain is not None:
                # Retrieve strain
                pass

            # Rotate back
            signal.samples_frequency = signal.xp.einsum(
                'rbpq,rbsq->rbsp',
                section_major_rotation.transpose((0, 1, 3, 2)),
                signal.samples_frequency,
                optimize = True
            )

        return signal

    @override
    def Jones(self, frequency_angular: (np.ndarray), verbose: bool = False) -> np.ndarray:
        assert len(frequency_angular.shape) == 1, f"frequency_angular must have shape [F,], but had shape {frequency_angular.shape}"
        
        if 'cupy' in sys.modules and isinstance(frequency_angular, cp.ndarray):
            xp = cp
            frequency_angular = cp.array(frequency_angular) # Ensure that the frequency array resides in the currently active cupy GPU
        else:
            xp = np

        section_DGDs = xp.array(self.section_DGD[:, :, None, None]) # Prepare extra dimensions for later calculations
        section_major_rotations = xp.array(rotation_matrix(self.section_major_angles)[:, :, None])
        section_birefringences = xp.array(self.section_birefringences[:, :, None, None])
        frequency_angular = frequency_angular[None, :, None]

        # iterable = section_matrices
        iterable = zip(section_DGDs, section_major_rotations, section_birefringences)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(
                iterable,
                total = self.section_count,
                desc = f"Building Jones matrix ({'CPU' if isinstance(frequency_angular, np.ndarray) else 'CUDA'})"
            )
        
        Jones_matrix = xp.tile(xp.eye(2, dtype = complex)[None, None], (self.realisation_count, frequency_angular.shape[1], 1, 1))
        for section_DGD, section_major_rotation, section_birefringence in iterable:
            # Rotate to local birefringence axes
            Jones_matrix = xp.einsum(
                'rspq,rsqw->rspw',
                section_major_rotation,
                Jones_matrix,
                optimize = True
            )

            # Apply differential phase
            differential_phase = xp.exp(-0.5j * (section_birefringence + section_DGD * frequency_angular * 1e-12))
            Jones_matrix *= xp.stack([differential_phase, differential_phase.conjugate()], axis = 2)

            # Rotate back
            Jones_matrix = xp.einsum(
                'rspq,rsqw->rspw',
                section_major_rotation.transpose((0, 1, 3, 2)),
                Jones_matrix,
                optimize = True
            )
            
        return Jones_matrix