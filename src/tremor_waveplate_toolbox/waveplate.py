"""
The waveplate fibre channel model for dual-polarisation transmission.
"""

from configparser import ConfigParser
from typing import override

import numpy as np
import scipy as sp

from .constants import PAULI_VECTOR, Domain
from .signal import Signal
from .channel import Channel

class Waveplate(Channel):
    """
    A class representing a waveplate model, which applies differential group delay and rotates state of polarisation in a distributed manner.
    Polarisation-dependent loss is neglected.
    Slow PMD drift and/or earthquake strain can be modelled.
    """
    def __init__(self, parameters: ConfigParser):
        """
        Create multiple waveplate channels for simultaneous propagation.

        Required entries in parameters['FIBRE']:
        - section_length [float]: correlation length in km.
        - section_count [int]: the number of fibre sections, each of which has length section_length.
        - PMD_parameter [float]: average accumulated differential group delay in ps/(km ^ 0.5).
        - realisation_count [int]: the number of fibre realisations with different distributed polarisation mode dispersion.
        - photoelasticity [float]: the fibre photoelasticity.
        """
        super().__init__(parameters)

        assert 'FIBRE'             in parameters, f"Parameters are missing section 'FIBRE'."
        assert 'section_length'    in parameters['FIBRE'], f"'section_length' is missing from parameters section 'FIBRE'."
        assert 'section_count'     in parameters['FIBRE'], f"'section_count' is missing from parameters section 'FIBRE'."
        assert 'PMD_parameter'     in parameters['FIBRE'], f"'PMD_parameter' is missing from parameters section 'FIBRE'."
        assert 'realisation_count' in parameters['FIBRE'], f"'realisation_count' is missing from parameters section 'FIBRE'."
        assert 'photoelasticity'   in parameters['FIBRE'], f"'photoelasticity' is missing from parameters section 'FIBRE'"

        self._section_length    = parameters.getfloat('FIBRE', 'section_length')
        self._section_count     = int(parameters.getfloat('FIBRE', 'section_count'))
        self._PMD_parameter     = parameters.getfloat('FIBRE', 'PMD_parameter')
        self._realisation_count = int(parameters.getfloat('FIBRE', 'realisation_count'))
        self._photoelasticity   = parameters.getfloat('FIBRE', 'photoelasticity')

        self.init_DGD()
        self.init_rotations()
        self.init_strain()

    def init_DGD(self):
        """
        Initialise random differential group delay per realisation and fibre section in ps.
        """
        # section_DGD_mean  = self.PMD_parameter * np.sqrt(self.section_length * np.pi / 8) # Czegledi et al. (2016): Polarization-Mode Dispersion Aware Digital Backpropagation, Prola et al. (1997): PMD Emulators and Signal Distortion in 2.48-Gb/s IM-DD Lightwave Systems
        section_DGD_mean  = self.PMD_parameter * np.sqrt(self.section_length * np.pi * 3 / 8) # Czegledi et al. (2016): Polarization-Mode Dispersion Aware Digital Backpropagation, Prola et al. (1997): PMD Emulators and Signal Distortion in 2.48-Gb/s IM-DD Lightwave Systems
        section_DGD_stdev = section_DGD_mean / 5
        self.section_DGD = np.random.default_rng().normal(
            loc = section_DGD_mean,
            scale = section_DGD_stdev,
            size = (self.section_count, self.realisation_count)
        )

    def init_rotations(self):
        """
        Initialise a random rotation of the state of polarisation per realisation and fibre section, in Stokes coordinates.
        """
        initialisation_vector = np.random.default_rng().normal(
            size = (self.section_count, self.realisation_count, 4)
        ) # Initialise [cos(theta), a * sin(theta)], Czegledi et al. (2016): Polarization Drift Channel Model for Coherent Fibre-Optic Systems
        initialisation_vector /= np.linalg.norm(initialisation_vector, axis = -1)[..., None] # Normalise

        rotation_angle = np.arccos(initialisation_vector[..., 0, None])
        rotation_axis  = initialisation_vector[..., 1:] / np.sin(rotation_angle)
        self.section_SOP_rotation_stokes = rotation_angle * rotation_axis # ???, notation error in Czegledi et al. ???

    def init_strain(self):
        """
        Initialise the fibre with 0 material strain.
        """
        self.section_material_strain = np.zeros(
            shape = (self.section_count),
            dtype = float
        )

    @override
    def propagate(self, signal: Signal, verbose: bool = False) -> Signal:
        """
        Propagate a polarisation-multiplexed phase-multiplexed signal through the waveplate model.
        The model applies a differential group delay and rotates the principal axes of polarisation in every fibre section.
        Multiple fibre realisations are applied at once.

        Inputs:
        - signal [Signal]: the signal to propagate through the channel in the time domain, shape [...,S,P] with sample count S and principal polarisations P=2.
        - verbose [bool]: whether to show a progress bar

        Outputs:
        - [Signal]: the output signal, shape [realisation_count,...,S,P] with number of realisations realisation_count
        """
        signal = signal.copy()
        signal.samples_frequency = signal.samples_frequency * np.ones(shape = (self.realisation_count, 1, 1, 1), dtype = int)
        frequency_angular = signal.frequency_angular[None, None]

        iterable = zip(self.section_DGD, self.section_SOP_rotation, self.section_optical_strain)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(iterable, total = self.section_count, desc = "Propagating signal through waveplate model")

        for section_DGD, section_SOP_rotation, section_optical_strain in iterable:
            # Apply section DGD
            section_DGD = section_DGD[:,None,None]
            DGD = np.exp(-0.5j * section_DGD * (1 + section_optical_strain) * frequency_angular * 1e-12) # [R, ..., 1] * [1, ..., S] = [R, ..., S]
            signal.samples_frequency[..., 0] *= DGD
            signal.samples_frequency[..., 1] *= np.conj(DGD)

            # Rotate PSP
            signal.samples_frequency = np.einsum('rpq,rbsq->rbsp', section_SOP_rotation, signal.samples_frequency)

        return signal

    def PMD_jones(self, w: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Calculate the frequency-dependent Jones matrix that describes this channel

        Inputs:
        - w [np.ndarray]: frequencies in Rad/s to calculate the Jones matrix for
        - verbose [bool]: whether to show a progress bar

        Outputs:
        - [np.ndarray]: Jones matrix of shape [realisation_count, W, 2, 2] where realisation_count is the number of fibre realisations and W is the length of w
        """
        assert len(w.shape) == 1, f"w should have one dimension, but had {len(w.shape)}"

        jones_matrix = np.zeros((self.realisation_count, len(w), 2, 2), dtype = complex)
        jones_matrix[:, :, (0, 1), (0, 1)] = 1

        iterable = zip(self.section_DGD, self.section_SOP_rotation, self.section_optical_strain)
        if verbose:
            from tqdm import tqdm
            iterable = tqdm(iterable, total = self.section_count, desc = "Calculating waveplate Jones matrix")

        for section_DGD, section_SOP_rotation, section_optical_strain in iterable:
            DGD = np.exp(-0.5j * section_DGD[:, None, None] * (1 + section_optical_strain) * w[None, :, None] * 1e-12)
            jones_matrix[:, :, 0, :] *= DGD
            jones_matrix[:, :, 1, :] *= np.conj(DGD)
            jones_matrix = section_SOP_rotation[:, None] @ jones_matrix

        return jones_matrix

    def to_dict(self) -> dict:
        """
        Represent this waveplate model (and its exact realisations) as a dictionary.

        Outputs:
        - [dict] The dictionary representation of this waveplate model
        """
        return {
            'section_length':              self.section_length,
            'section_count':               self.section_count,
            'PMD_parameter':               self.PMD_parameter,
            'realisation_count':           self.realisation_count,
            'photoelasticity':             self.photoelasticity,
            'section_DGD':                 self.section_DGD,
            'section_SOP_rotation_stokes': self.section_SOP_rotation_stokes,
            'section_material_strain':     self.section_material_strain,
        }

    @classmethod
    def from_dict(cls, waveplate_dict: dict):
        """
        Create a waveplate model from a saved dictionary.

        Inputs:
        - waveplate_dict [dict]: a dictionary created using Waveplate.to_dict()

        Outputs:
        - [Waveplate] the loaded waveplate model instance.
        """
        parameters = ConfigParser()
        parameters.add_section('FIBRE')
        parameters.set('FIBRE', 'section_length', waveplate_dict['section_length'])
        parameters.set('FIBRE', 'section_count', waveplate_dict['section_count'])
        parameters.set('FIBRE', 'PMD_parameter', waveplate_dict['PMD_parameter'])
        parameters.set('FIBRE', 'realisation_count', waveplate_dict['realisation_count'])
        parameters.set('FIBRE', 'photoelasticity', waveplate_dict['photoelasticity'])

        waveplate = cls(parameters)
        waveplate.section_DGD = waveplate_dict['section_DGD']
        waveplate.section_SOP_rotation_stokes = waveplate_dict['section_SOP_rotation_stokes']
        waveplate.section_material_strain = waveplate_dict['section_material_strain']

        return waveplate

    @property
    def section_length(self) -> float:
        """
        [float] correlation length of this waveplate model in km.
        """
        return float(self._section_length)

    @section_length.setter
    def section_length(self, value):
        raise AttributeError("The correlation length section_length cannot be changed after creation of the waveplate model.")

    @property
    def section_count(self) -> int:
        """
        [int] number of fibre sections of this waveplate model.
        """
        return int(self._section_count)

    @section_count.setter
    def section_count(self, value):
        raise AttributeError("The section count section_count cannot be changed after creation of the waveplate model.")

    @property
    def PMD_parameter(self) -> float:
        """
        [float] polarisation mode dispersion parameter of this waveplate model in ps/(km ^ 0.5)
        """
        return float(self._PMD_parameter)

    @PMD_parameter.setter
    def PMD_parameter(self, value):
        raise AttributeError("The polarisation mode dispersion parameter PMD_parameter cannot be changed after creation of the waveplate model")

    @property
    def realisation_count(self) -> float:
        """
        [int] realisation count of this waveplate model
        """
        return int(self._realisation_count)

    @realisation_count.setter
    def realisation_count(self, value):
        raise AttributeError("The realisation count realisation_count cannot be changed after creation of the waveplate model")

    @property
    def length(self) -> float:
        """
        [float] total length of the fibre in km
        """
        return float(self.section_length * self.section_count)

    @length.setter
    def length(self, value):
        raise AttributeError("The fibre length cannot be set directly")

    @property
    def photoelasticity(self) -> float:
        """
        [float] fibre photoelasticity constant
        """
        return self._photoelasticity

    @photoelasticity.setter
    def photoelasticity(self, value):
        raise AttributeError("The photoelasticity constant cannot be changed after creation of the waveplate model")

    @property
    def section_DGD(self) -> np.ndarray:
        """
        [float] the differential group delay per section and realisation in ps, shape [section_count, realisation_count]
        """
        return self._section_DGD

    @section_DGD.setter
    def section_DGD(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), f"New section_DGD must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New section_DGD array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.section_count, self.realisation_count), f"New section_DGD array must have shape (self.section_count ({self.section_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        self._section_DGD = value.copy().astype(float)

    @property
    def DGD(self) -> np.ndarray:
        """
        Accumulated differential group delay in ps, shape [realisation_count] where realisation_count is the number of fibre realisations
        """
        w = np.array([-2e10, 2e10])

        jones_matrices = self.PMD_jones(w)
        jones_matrices_derivative = np.diff(jones_matrices, axis = 1)[:, 0] / np.diff(w)[0]

        accumulated_DGD = 2 * np.sqrt(np.linalg.det(jones_matrices_derivative)) * 1e12 # Gordon et al. - PMD Fundamentals: Polarization Mode Dispersion in Optical Fibers
        accumulated_DGD = accumulated_DGD.real.astype(float)

        return accumulated_DGD

    @DGD.setter
    def DGD(self, value):
        raise AttributeError("The accumulated DGD cannot be set directly; set section DGD instead")

    @property
    def section_SOP_rotation_stokes(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] state of polarisation rotations per section, expressed in Stokes parameters.
        """
        return self._section_SOP_rotation_stokes

    @section_SOP_rotation_stokes.setter
    def section_SOP_rotation_stokes(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), f"New section_SOP_rotation_stokes value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New section_SOP_rotation_stokes array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.section_count, self.realisation_count, 3), f"New section_SOP_rotation_stokes array must have shape (self.section_count ({self.section_count}), self.realisation_count ({self.realisation_count}), 3), but had shape {value.shape}"
        self._section_SOP_rotation_stokes = value.copy().astype(float)
        self._section_SOP_rotation = sp.linalg.expm(-1j * np.tensordot(self.section_SOP_rotation_stokes, PAULI_VECTOR, 1))

    @property
    def section_SOP_rotation(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] state of polarisation rotations per section, expressed using rotation matrices
        """
        return self._section_SOP_rotation

    @section_SOP_rotation.setter
    def section_SOP_rotation(self, value):
        raise AttributeError("The state of polarisation rotations cannot be set as rotation matrices; set section_SOP_rotation_stokes instead")
        # assert isinstance(value, np.ndarray), f"New PSP value must be type np.ndarray, but was a {type(value)}"
        # assert value.dtype in (float, int, comples), f"New PSP array must contain values of type complex, but contained {value.dtype}"
        # assert value.shape == (self.Nsec, self.Nreal, 2, 2), f"New PSP array must have shape (self.Nsec ({self.Nsec}), self.Nreal ({self.Nreal}), 2, 2), but had shape {value.shape}"
        # new_PSP = value.copy().astype(complex)
        #
        # new_PSP_stokes = np.zeros(shape = (self.Nsec, self.Nreal, 3), dtype = complex)
        # new_PSP_log = sp.linalg.logm(new_PSP) / -1j # np.tensordot(self.PSP_stokes, PAULI_VECTOR, 1
        # assert np.allclose(PSP_log[:, :, 0, 0], -PSP_log[:, :, 1, 1]) and np.allclose(PSP_log[:, :, (0, 1), (0, 1)], PSP_log[:, :, (0, 1), (0, 1)].real), f"No PAULI_1 scalar can be found from new PSP matrix"
        # new_PSP_stokes[:, :, 0] = (PSP_log[:, :, 0, 0] - PSP_log[:, :, 1, 1]).real / 2
        # assert np.allclose(PSP_log[:, :, 0, 1].real, PSP_log[:, :, 1, 0].real), f"No PAULI_2 scalar can be found from new PSP matrix"
        # new_PSP_stokes[:, :, 1] = (PSP_log[:, :, 0, 1] + PSP_log[:, :, 1, 0]).real / 2
        # assert np.allclose(PSP_log[:, :, 0, 1].imag, PSP_log[:, :, 1, 0].imag), f"No PAULI_3 scalar can be found from new PSP matrix"
        # new_PSP_stokes[:, :, 2] = -(PSP_log[:, :, 0, 1] - PSP_log[:, :, 1, 0]).imag / 2
        # new_PSP_stokes = new_PSP_stokes.real.astype(float)
        #
        # self._PSP = new_PSP
        # self._PSP_stokes = new_PSP_stokes

    @property
    def section_material_strain(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] material strain parameter per fibre section.
        """
        return self._section_material_strain

    @section_material_strain.setter
    def section_material_strain(self, value):
        assert isinstance(value, np.ndarray), f"New section_material_strain value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New section_material_strain array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.section_count,), f"New section_material_strain array must have shape (self.section_count ({self.section_count}),), but had shape {value.shape}"
        self._section_material_strain = value.copy().astype(float)

    @property
    def section_optical_strain(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] optical strain parameter per fibre section.
        """
        return self.photoelasticity * self.section_material_strain

    @section_optical_strain.setter
    def section_optical_strain(self, value):
        raise AttributeError("The optical strain cannot be set directly; set section_material_strain instead")
