"""
An optical fibre channel model base class for dual-polarisation transmission, based on Marcuse's method.
Currently it models only PMD effects: (earthquake-dependent) differential group delay and state of polarisation scramblers.
"""
from typing import override

import numpy as np

from .fibre import Fibre
from .signal import Signal
from .earthquake import Earthquake
from .utils import rotation_matrix
from .constants import Device

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
        self.section_DGD = np.tile(2 * self.PMD_parameter / np.sqrt(8 * self.correlation_length) * self.section_path.lengths[:, None], (1, self.realisation_count))

    @override
    def _init_PSP(self):
        """
        In Marcuse's method, PSPs of subsequent sections are strongly correlated, and DGD is distributed uniformly along the fibre.
        """
        self._section_birefringences = 0
        self.section_birefringences  = np.tile(2 * np.pi / self.beat_length * self.section_path.lengths[:, None], (1, self.realisation_count))
        self._section_major_angles   = 0
        self.section_major_angles    = np.cumsum(np.sqrt(self.section_path.lengths / (2 * self.correlation_length))[:, None] * np.random.default_rng().normal(size = (self.section_path.edge_count, self.realisation_count)), axis = 0)

    @override
    def _propagate_master(self, signal: Signal, frequency_angular: np.ndarray, transmission_start_time: float = 0, earthquake: Earthquake = None, earthquake_batch_size: int = 100, verbose: bool = False) -> Signal:
        """
        Master function both for propagating a signal or building a Jones transfer matrix
        """
        if signal.device == Device.CUDA: signal.to_device(Device.CUDA) # Ensure that the signal resides in the currently active cupy GPU
        
        section_DGDs = signal.xp.array(self.section_DGD[:, :, None, None]) # [S, R, 1, 1]
        section_major_rotations = signal.xp.array(rotation_matrix(self.section_major_angles[:, :, None])) # [S, R, 1, 2, 2]
        section_birefringences = signal.xp.array(self.section_birefringences[:, :, None, None]) # [S, R, 1, 1]

        signal = signal.copy() # [R, B, F, 2]/[R, F, 2, 2]
        frequency_angular = frequency_angular[*(None,) * signal.sample_axis_nonnegative, :, *(None,) * (2 - signal.sample_axis_nonnegative)] # [1, 1, F]/[1, F, 1]

        iterable = zip(section_DGDs, section_major_rotations, section_birefringences)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(
                iterable,
                total = self.section_path.edge_count,
                desc = f"{"Propagating signal through fibre" if signal.sample_axis_negative == -2 else "Building Jones matrix"} ({'CPU' if signal.device == Device.CPU else 'CUDA'}{', perturbed' if earthquake is not None else ''})"
            )

        for section_index, (section_DGD, section_major_rotation, section_birefringence) in enumerate(iterable): # [R, 1, 1], [R, 1, 2, 2], [R, 1, 1]
            # Request earthquake strain for the next sections, if necessary
            if earthquake is not None and section_index % earthquake_batch_size == 0:
                _, _, _, strain = earthquake(self.section_path[section_index * earthquake_batch_size:(section_index + 1) * earthquake_batch_size + 1], verbose = verbose)
                strain.to_device(signal.device)
                strain_start_section_index = section_index
                
            # Rotate to local birefringence axes
            signal.samples_frequency = signal.xp.einsum( # [R, 1, 2, 2] @ [R, B, F, 2]/[R, F, 2, 2] = [R, B, F, 2]/[R, F, 2, 2]
                'rbpq,rbsq->rbsp',
                section_major_rotation,
                signal.samples_frequency,
                optimize = True
            )
            
            # Apply differential phase
            differential_phase = section_birefringence # [R, 1, 1]
            if self.PMD_parameter != 0: differential_phase = differential_phase + section_DGD * frequency_angular * 1e-12 # [R, 1, 1] + [R, 1, 1] * [1, 1, F]/[1, F, 1] = [R, 1, F]/[R, F, 1]
            if earthquake is not None:
                time = transmission_start_time + self.section_path.centre_positions[section_index] / self.group_velocity(signal.carrier_wavelength) # 1
                section_strain = strain.samples_time[section_index - strain_start_section_index, int(np.floor(time * strain.sample_rate))] # 1
                differential_phase = differential_phase * (1 + self.photoelasticity * section_strain) # [R, 1, F]/[R, F, 1] * 1 = [R, 1, F]/[R, F, 1]

            differential_phase = signal.xp.exp(-0.5j * differential_phase) # [R, 1, F]/[R, F, 1]
            signal.samples_frequency = signal.samples_frequency * signal.xp.stack([differential_phase, differential_phase.conjugate()], axis = 3) # [R, B, F, 2]/[R, F, 2, 2] * [R, 1, F, 2]/[R, F, 1, 2] = [R, B, F, 2]/[R, F, 2, 2]

            # Rotate back
            signal.samples_frequency = signal.xp.einsum( # [R, 1, 2, 2] @ [R, B, F, 2]/[R, F, 2, 2] = [R, B, F, 2]/[R, F, 2, 2]
                'rbpq,rbsq->rbsp',
                section_major_rotation.transpose((0, 1, 3, 2)),
                signal.samples_frequency,
                optimize = True
            )

        return signal

    @override
    def to_dict(self):
        return super().to_dict() | {
                'section_major_angles':   self.section_major_angles.tolist(),
                'section_birefringences': self.section_birefringences.tolist()
            }

    @classmethod
    @override
    def from_dict(cls, fibre_dict: dict):
        fibre = super(FibreMarcuse, cls).from_dict(fibre_dict)
        fibre.section_major_angles   = np.array(fibre_dict['section_major_angles'])
        fibre.section_birefringences = np.array(fibre_dict['section_birefringences'])
        return fibre

    @override
    def __eq__(self, other) -> bool:
        return super().__eq__(other) and \
            np.all(self._section_major_angles   == other._section_major_angles)  and \
            np.all(self._section_birefringences == other._section_birefringences)

    @property
    def section_major_angles(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] orientation of the major axes of birefringence per section in radians
        """
        return self._section_major_angles

    @section_major_angles.setter
    def section_major_angles(self, value) -> None:
        assert self.section_major_angles is not None, f"Fibre was initialised with PSP matrices (not angles), and this cannot be changed"
        assert isinstance(value, np.ndarray), f"New section_major_angles value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New section_major_angles array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.section_path.edge_count, self.realisation_count), f"New section_major_angles array must have shape (self.section_path.edge_count ({self.section_path.edge_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        # assert np.all((value >= -np.pi) & (value < np.pi)), f"New section_major_angles array must have values between -pi and pi"
        self._section_major_angles = value.copy().astype(float)

    @property
    def section_birefringences(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] local differential phase between the major polarisation axes per section in ps
        """
        return self._section_birefringences

    @section_birefringences.setter
    def section_birefringences(self, value) -> None:
        assert self.section_birefringences is not None, f"Fibre was initialised with PSP matrices (not angles), and this cannot be changed"
        assert isinstance(value, np.ndarray), f"New section_birefringences value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New section_birefringences array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.section_path.edge_count, self.realisation_count), f"New section_birefringences array must have shape (self.section_path.edge_count ({self.section_path.edge_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        # assert np.all((value >= -np.pi) & (value < np.pi)), f"New section_birefringences array must have values between -pi and pi"
        self._section_birefringences = value.copy().astype(float)