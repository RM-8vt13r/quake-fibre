"""
An optical fibre channel model base class for dual-polarisation transmission, based on the coarse-step method.
Currently it models only PMD effects: (earthquake-dependent) differential group delay and state of polarisation scramblers.
"""
from configparser import ConfigParser
from typing import override
import sys

import numpy as np
import scipy as sp
try:
    import cupy as cp
except:
    pass

from .fibre import Fibre
from .signal import Signal
from .constants import PAULI_VECTOR, Device

class FibreCoarseStep(Fibre):
    """
    A class representing an optical fibre.
    Currently it models only polarisation-mode dispersion, using the coarse-step method.
    This model applies differential group delay and scrambles the state of polarisation in a completely random and distributed manner.
    Chromatic dispersion, the Kerr effect, attenuation, EDFA noise, and polarisation-dependent loss are neglected, and slow PMD drift are not implemented.
    Earthquake strain can be modelled as a change in differential phase.
    """
    @override
    def _init_DGD(self):
        """
        In the coarse-step method, DGD is drawn randomly per section from a Gaussian distribution.
        """
        section_DGD_means  = self.PMD_parameter * np.sqrt(self.section_lengths * np.pi * 3 / 8) # Czegledi et al. (2016): Polarization-Mode Dispersion Aware Digital Backpropagation, Prola et al. (1997): PMD Emulators and Signal Distortion in 2.48-Gb/s IM-DD Lightwave Systems
        
        section_DGD_stdevs = section_DGD_means / 5
        self.section_DGD = np.random.default_rng().normal(
            loc   = section_DGD_means[:, None],
            scale = section_DGD_stdevs[:, None],
            size  = (self.section_count, self.realisation_count)
        )

    @override
    def _init_PSP(self):
        """
        In the coarse-step method, SOP is scrambled random uniformly over the Poincaré sphere after each section.
        """
        initialisation_vector = np.random.default_rng().normal(
           size = (self.section_count, self.realisation_count, 4)
        ) # Initialise [cos(theta), a * sin(theta)], Czegledi et al. (2016): Polarization Drift Channel Model for Coherent Fibre-Optic Systems
        initialisation_vector /= np.linalg.norm(initialisation_vector, axis = -1)[..., None] # [S, R, 4]

        rotation_angle    = np.arccos(initialisation_vector[..., 0]) # [S, R]
        rotation_axis     = initialisation_vector[..., 1:] / np.sin(rotation_angle)[..., None] # [S, R, 3]
        self._section_PSP = 0
        self.section_PSP  = sp.linalg.expm(-1j * rotation_angle[:, :, None, None] * np.einsum('sra,apq->srpq', rotation_axis, PAULI_VECTOR))
        self._section_major_angles   = None
        self._section_birefringences = None

    @override
    def propagate(self, signal: Signal, strain: Signal = None, verbose: bool = False) -> np.ndarray:
        """
        This model applies a differential group delay and scrambles the state of polarisation randomly in every fibre section.
        """
        assert strain is None, f"earthquake functionality not implemented yet"

        if signal.device == Device.CUDA: signal.to_device(Device.CUDA) # Ensure that the signal resides in the currently active cupy GPU

        section_DGDs = signal.xp.array(self.section_DGD)
        section_PSPs = signal.xp.array(self.section_PSP)

        signal = signal.copy()
        signal.samples_frequency = signal.xp.tile(signal.samples_frequency, (self.realisation_count, 1, 1, 1))
        frequency_angular = signal.frequency_angular[None, None]

        iterable = zip(section_DGDs, section_PSPs)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(
                iterable,
                total = self.section_count,
                desc = f"Propagating signal through fibre ({'CPU' if signal.device == Device.CPU else 'CUDA'})"
            )

        for section_DGD, section_PSP in iterable:
            # Apply section DGD
            DGD = signal.xp.exp(-0.5j * section_DGD[:, None, None] * frequency_angular * 1e-12)
            signal.samples_frequency[..., 0] *= DGD
            signal.samples_frequency[..., 1] *= DGD.conjugate()

            # Scramble SOP
            signal.samples_frequency = signal.xp.einsum(
                'rpq,rbsq->rbsp',
                section_PSP,
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

        section_DGDs = xp.array(self.section_DGD[:, :, None, None])
        section_PSPs = xp.array(self.section_PSP)
        frequency_angular = frequency_angular[None, :, None] # Prepare extra dimensions for later calculations

        iterable = zip(section_DGDs, section_PSPs)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(
                iterable,
                total = self.section_count,
                desc = f"Building Jones matrix ({'CPU' if isinstance(frequency_angular, np.ndarray) else 'CUDA'})"
            )
        
        Jones_matrix = xp.tile(xp.eye(2, dtype = complex)[None, None], (self.realisation_count, frequency_angular.shape[1], 1, 1))
        for section_DGD, section_PSP in iterable:
            # Apply section DGD
            differential_phase = xp.exp(-0.5j * section_DGD * frequency_angular * 1e-12)
            
            Jones_matrix[:, :, 0, :] *= differential_phase
            Jones_matrix[:, :, 1, :] *= differential_phase.conjugate()

            # Scramble SOP
            Jones_matrix = xp.einsum(
                'rpq,rfqs->rfps',
                section_PSP,
                Jones_matrix,
                optimize = True
             )

        return Jones_matrix