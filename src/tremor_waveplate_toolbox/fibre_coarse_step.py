"""
An optical fibre channel model base class for dual-polarisation transmission, based on the coarse-step method.
Currently it models only PMD effects: (perturbed) differential phase and major birefringence axes rotations.
"""
from typing import override

import numpy as np
import scipy as sp

from .fibre import Fibre
from .signal import Signal
from .perturbation import Perturbation
from .constants import Device, PAULI_VECTOR

class FibreCoarseStep(Fibre):
    """
    A class representing an optical fibre.
    Currently it models only polarisation-mode dispersion, using the coarse-step method.
    This model applies differential group delay and scrambles the state of polarisation in a completely random and distributed manner.
    Chromatic dispersion, the Kerr effect, attenuation, EDFA noise, and polarisation-dependent loss are neglected, and slow PMD drift are not implemented.
    External perturbations can be modelled as a change in differential phase or major birefringence axes orientations.
    """
    @override
    def _init_DGD(self):
        """
        In the coarse-step method, DGD is drawn randomly per section from a Gaussian distribution.
        """
        section_DGD_means  = self.PMD_parameter * np.sqrt(self.section_path.lengths * np.pi * 3 / 8) # Czegledi et al. (2016): Polarization-Mode Dispersion Aware Digital Backpropagation, Prola et al. (1997): PMD Emulators and Signal Distortion in 2.48-Gb/s IM-DD Lightwave Systems
        
        section_DGD_stdevs = section_DGD_means / 5
        self.section_DGD = np.random.default_rng().normal(
            loc   = section_DGD_means[:, None],
            scale = section_DGD_stdevs[:, None],
            size  = (self.section_path.edge_count, self.realisation_count)
        )

    @override
    def _init_PSP(self):
        """
        In the coarse-step method, SOP is scrambled random uniformly over the Poincaré sphere after each section.
        """
        initialisation_vector = np.random.default_rng().normal(
           size = (self.section_path.edge_count, self.realisation_count, 4)
        ) # Initialise [cos(theta), a * sin(theta)], Czegledi et al. (2016): Polarization Drift Channel Model for Coherent Fibre-Optic Systems
        initialisation_vector /= np.linalg.norm(initialisation_vector, axis = -1)[..., None]

        rotation_angle    = np.arccos(initialisation_vector[..., 0])
        rotation_axis     = initialisation_vector[..., 1:] / np.sin(rotation_angle)[..., None]
        self._section_PSP = 0
        self.section_PSP  = sp.linalg.expm(-1j * rotation_angle[:, :, None, None] * np.einsum('sra,apq->srpq', rotation_axis, PAULI_VECTOR))
        
    @override
    def _propagate_master(self, signal: Signal, frequency_angular: np.ndarray, transmission_start_times: (float, np.ndarray) = 0, perturbation: Perturbation = None, verbose: bool = False) -> Signal:
        """
        Master function both for propagating a signal or building a Jones transfer matrix
        """
        assert perturbation is None, f"Perturbation not yet implemented in the coarse-step fibre model"

        if not isinstance(transmission_start_times, (float, int)):
            transmission_start_times = signal.xp.array(transmission_start_times)
            assert len(transmission_start_times.shape) == 1, f"transmission_start_times must have shape [T,], but had shape {transmission_start_times.shape}"
            assert signal.shape[signal.sample_axis - 1] == 1, f"If transmission_start_times has shape [T,] signal must have batch size 1, but this was {signal.shape[signal.sample_axis - 1]}"
        else:
            transmission_start_times = signal.xp.array([transmission_start_times])

        section_DGDs = signal.xp.array(self.section_DGD[:, :, *(None,) * -signal.sample_axis_negative]) # [S, R, 1, 1]/[S, R, 1, 1, 1]
        section_PSPs = signal.xp.array(self.section_PSP[:, :, *(None,) * -(1 + signal.sample_axis_negative)]) # [S, R, 1, 2, 2]/[S, R, 1, 1, 2, 2]

        signal = signal.copy()
        frequency_angular = frequency_angular[*(None,) * 2, :, *(None,) * -(2 + signal.sample_axis_negative)] # [1, 1, F]/[1, 1, F, 1]

        iterable = zip(section_DGDs, section_PSPs)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(
                iterable,
                total = self.section_path.edge_count,
                desc = f"{"Propagating signal through fibre" if signal.sample_axis_negative == -2 else "Building Jones matrix"} ({'CPU' if signal.device == Device.CPU else 'CUDA'}{', perturbed' if perturbation is not None else ''})"
            )

        for section_DGD, section_PSP in iterable: # [R, 1, 1], [R, 1]
            if self.PMD_parameter != 0:
                # Apply section DGD
                differential_phase = signal.xp.exp(-0.5j * section_DGD * frequency_angular * 1e-12) # [R, 1, 1]/[R, 1, 1, 1] * [1, 1, F]/[1, 1, F, 1] = [R, 1, F]/[R, 1, F, 1]
                signal.samples_frequency = signal.samples_frequency * signal.xp.stack([differential_phase, differential_phase.conjugate()], axis = -1) # [R, B, F, 2]/[R, B, F, 2, 2] * [R, 1, F, 2]/[R, 1, F, 1, 2] = [R, B, F, 2]/[R, B, F, 2, 2]

            # Scramble SOP
            signal.samples_frequency = signal.xp.einsum( # [R, 1, 2, 2]/[R, 1, 1, 2, 2] @ [R, B, F, 2]/[R, B, F, 2, 2] = [R, B, F, 2]/[R, B, F, 2, 2]
                '...pq,...sq->...sp',
                section_PSP,
                signal.samples_frequency,
                optimize = True
            )

        return signal

    @override
    def to_dict(self):
        return super().to_dict() | {
                'section_PSP': self.section_PSP.tolist()
            }

    @classmethod
    @override
    def from_dict(cls, fibre_dict: dict):
        fibre = super(FibreCoarseStep, cls).from_dict(fibre_dict)
        fibre.section_PSP = np.array(fibre_dict['section_PSP'])
        return fibre

    @override
    def __eq__(self, other) -> bool:
        return super().__eq__(other) and \
            np.all(self._section_PSP == other._section_PSP)

    @property
    def section_PSP(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] matrices that rotate the state of polarisation (SOP) to the local principle states of polarisation (PSPs)
        """
        return self._section_PSP

    @section_PSP.setter
    def section_PSP(self, value) -> None:
        assert self._section_PSP is not None, f"Fibre was initialised with PSP angles (not matrices), and this cannot be changed"
        assert isinstance(value, np.ndarray), f"New section_PSP value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int, complex), f"New section_PSP array must contain values of type complex, but contained {value.dtype}"
        assert value.shape == (self.section_path.edge_count, self.realisation_count, 2, 2), f"New section_PSP array must have shape (self.section_path.edge_count ({self.section_path.edge_count}), self.realisation_count ({self.realisation_count}), 2, 2), but had shape {value.shape}"
        assert np.allclose(value[..., 0, 0], value[..., 1, 1].conjugate()) and np.allclose(value[..., 1, 0], -value[..., 0, 1].conjugate()), f"New section_PSP array must contain unitary matrices, but didn't"
        self._section_PSP = value.copy().astype(complex)