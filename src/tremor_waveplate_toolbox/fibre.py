"""
An optical fibre channel model for dual-polarisation transmission.
Currently it models only PMD effects: (earthquake-dependent) differential group delay and state of polarisation scramblers.
"""

from configparser import ConfigParser
import json

import numpy as np
import scipy as sp
import obspy as op

from .constants import PAULI_VECTOR, Domain
from .signal import Signal

class Fibre:
    """
    A class representing an optical fibre.
    Currently it consists only of the waveplate model, which applies differential group delay and rotates state of polarisation in a distributed manner.
    Chromatic dispersion, the Kerr effect, attenuation, EDFA noise, polarisation-dependent loss is neglected, and slow PMD drift are not implemented.
    Earthquake strain can be modelled as a change in differential group delay.
    """
    def __init__(self, parameters: ConfigParser):
        """
        Instantiate multiple fibre channels for simultaneous propagation.

        Required entries in parameters['FIBRE']:
        - section_length [float]:  correlation length in km.
        - path_coordinates [list]: list of coordinates (longitude, latitude) along the fibre path.
        - PMD_parameter [float]:   average accumulated differential group delay in ps/(km ^ 0.5).
        - realisation_count [int]: the number of fibre realisations with different distributed polarisation mode dispersion.
        - photoelasticity [float]: the fibre photoelasticity.
        """
        assert 'FIBRE'             in parameters, f"Parameters are missing section 'FIBRE'."
        assert 'section_length'    in parameters['FIBRE'], f"'section_length' is missing from parameters section 'FIBRE'."
        assert 'PMD_parameter'     in parameters['FIBRE'], f"'PMD_parameter' is missing from parameters section 'FIBRE'."
        assert 'realisation_count' in parameters['FIBRE'], f"'realisation_count' is missing from parameters section 'FIBRE'."
        assert 'photoelasticity'   in parameters['FIBRE'], f"'photoelasticity' is missing from parameters section 'FIBRE'"
        assert 'path_coordinates' in parameters['FIBRE'] or 'section_count' in parameters['FIBRE'], f"Parameters section 'FIBRE' must contain variable 'path_coordinates' or 'section_count'."

        self._section_length    = parameters.getfloat('FIBRE', 'section_length')
        self._PMD_parameter     = parameters.getfloat('FIBRE', 'PMD_parameter')
        self._realisation_count = int(parameters.getfloat('FIBRE', 'realisation_count'))
        self._photoelasticity   = parameters.getfloat('FIBRE', 'photoelasticity')
        
        if 'path_coordinates' in parameters['FIBRE']:
            self._path_coordinates = np.array(json.loads(parameters.get('FIBRE', 'path_coordinates')), dtype = float)
            self._section_count    = None
        else:
            self._path_coordinates = None
            self._section_count    = parameters.getint('FIBRE', 'section_count')

        self._init_path()
        self._init_DGD()
        self._init_SOP_rotations()
        self._init_material_strain()

    def _init_path(self):
        """
        Initialise fibre- and section path information.
        """
        if self._path_coordinates is not None:
            self._path_lengths = np.array([
                op.geodetics.base.calc_vincenty_inverse(*reversed(coordinate1), *reversed(coordinate2))[0] / 1000
                for coordinate1, coordinate2 in zip(self.path_coordinates[:-1], self.path_coordinates[1:])
            ])

            self._path_positions    = np.append([0], np.cumsum(self.path_lengths))
            self._section_positions = np.append(np.arange(0, self.path_positions[-1], self._section_length), self.path_positions[-1])
            
            section_longitudes = np.interp(self.section_positions, self.path_positions, self.path_coordinates[:, 0])
            section_latitudes  = np.interp(self.section_positions, self.path_positions, self.path_coordinates[:, 1])
            self._section_coordinates = np.stack([section_longitudes, section_latitudes], axis = 1)

            self._section_lengths = np.diff(self.section_positions)

            return

        self._path_lengths        = None
        self._path_positions      = None
        self._section_coordinates = None
        self._section_lengths     = np.full(
            shape      = (self._section_count,),
            fill_value = self._section_length,
            dtype      = float
        )
        self._section_positions   = np.append([0], np.cumsum(self.section_lengths))
        
    def _init_DGD(self):
        """
        Initialise random differential group delay per realisation and fibre section in ps.
        """
        section_DGD_means  = self.PMD_parameter * np.sqrt(self.section_lengths * np.pi * 3 / 8) # Czegledi et al. (2016): Polarization-Mode Dispersion Aware Digital Backpropagation, Prola et al. (1997): PMD Emulators and Signal Distortion in 2.48-Gb/s IM-DD Lightwave Systems
        section_DGD_stdevs = section_DGD_means / 5
        self.section_DGD = np.random.default_rng().normal(
            loc   = section_DGD_means[:, None],
            scale = section_DGD_stdevs[:, None],
            size  = (self.section_count, self.realisation_count)
        )

    def _init_SOP_rotations(self):
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

    def _init_material_strain(self):
        """
        Initialise the fibre with 0 material strain.
        """
        self.section_material_strain = np.zeros(
            shape = (self.section_count),
            dtype = float
        )

    def __call__(self, signal: Signal, verbose: bool = False) -> Signal:
        """
        Make fibre instances callable; see propagate()
        """
        return self.propagate(signal, verbose)

    def propagate(self, signal: Signal, verbose: bool = False) -> Signal:
        """
        Propagate a polarisation-multiplexed phase-multiplexed signal through the fibre.
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
            iterable = tqdm(
                iterable,
                total = self.section_count,
                desc = "Propagating signal through fibre"
            )

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
            iterable = tqdm(iterable, total = self.section_count, desc = "Calculating waveplate model Jones matrix")

        for section_DGD, section_SOP_rotation, section_optical_strain in iterable:
            DGD = np.exp(-0.5j * section_DGD[:, None, None] * (1 + section_optical_strain) * w[None, :, None] * 1e-12)
            jones_matrix[:, :, 0, :] *= DGD
            jones_matrix[:, :, 1, :] *= np.conj(DGD)
            jones_matrix = section_SOP_rotation[:, None] @ jones_matrix

        return jones_matrix

    def to_dict(self) -> dict:
        """
        Represent this fibre (and its exact realisations) as a dictionary.

        Outputs:
        - [dict] The dictionary representation of this fibre
        """
        fibre_dict = {
            'section_lengths':             self.section_lengths.tolist(),
            'section_positions':           self.section_positions.tolist(),
            'PMD_parameter':               self.PMD_parameter,
            'realisation_count':           self.realisation_count,
            'photoelasticity':             self.photoelasticity,
            'section_DGD':                 self.section_DGD.tolist(),
            'section_SOP_rotation_stokes': self.section_SOP_rotation_stokes.tolist(),
            'section_material_strain':     self.section_material_strain.tolist(),
        }

        if self._path_coordinates is not None:
            fibre_dict = fibre_dict | {
                'path_coordinates':    self.path_coordinates.tolist(),
                'path_lengths':        self.path_lengths.tolist(),
                'path_positions':      self.path_positions.tolist(),
                'section_coordinates': self.section_coordinates.tolist()
            }

        return fibre_dict

    @classmethod
    def from_dict(cls, fibre_dict: dict):
        """
        Instantiate a fibre from a saved dictionary.

        Inputs:
        - fibre_dict [dict]: a dictionary created using Fibre.to_dict()

        Outputs:
        - [Fibre] the loaded fibre instance.
        """
        parameters = ConfigParser()
        parameters.add_section('FIBRE')
        parameters.set('FIBRE', 'section_length',    str(fibre_dict['section_lengths'][0]))
        parameters.set('FIBRE', 'PMD_parameter',     str(fibre_dict['PMD_parameter']))
        parameters.set('FIBRE', 'realisation_count', str(fibre_dict['realisation_count']))
        parameters.set('FIBRE', 'photoelasticity',   str(fibre_dict['photoelasticity']))

        if 'path_coordinates' in fibre_dict:
            parameters.set('FIBRE', 'path_coordinates', str(fibre_dict['path_coordinates']))
        else:
            parameters.set('FIBRE', 'section_count', str(len(fibre_dict['section_lengths'])))

        fibre = cls(parameters)
        fibre.section_DGD                 = np.array(fibre_dict['section_DGD'])
        fibre.section_SOP_rotation_stokes = np.array(fibre_dict['section_SOP_rotation_stokes'])
        fibre.section_material_strain     = np.array(fibre_dict['section_material_strain'])

        return fibre

    def __eq__(self, other):
        if self._PMD_parameter                       == other._PMD_parameter                and \
            self._photoelasticity                    == other._photoelasticity              and \
            self._realisation_count                  == other._realisation_count            and \
            self._section_count                      == other._section_count                and \
            self._section_length                     == other._section_length               and \
            np.all(self._path_coordinates            == other._path_coordinates)            and \
            np.all(self._path_lengths                == other._path_lengths)                and \
            np.all(self._path_positions              == other._path_positions)              and \
            np.all(self._section_coordinates         == other._section_coordinates)         and \
            np.all(self._section_lengths             == other._section_lengths)             and \
            np.all(self._section_positions           == other._section_positions)           and \
            np.all(self._section_DGD                 == other._section_DGD)                 and \
            np.all(self._section_SOP_rotation_stokes == other._section_SOP_rotation_stokes) and \
            np.all(self._section_material_strain     == other._section_material_strain):
            return True

        return False

    @property
    def path_coordinates(self) -> np.ndarray:
        """
        [np.ndarray] array of fibre path coordinates, shape [S+1, 2] where S is the number of path segments and the second dimension contains longitude, latitude
        """
        if self._path_coordinates is not None:
            return self._path_coordinates
        raise AttributeError("Path coordinates were not passed to Fibre constructor, and are therefore not available")

    @path_coordinates.setter
    def path_coordinates(self, value):
        raise AttributeError("The path coordinates path_coordinates cannot be changed after instantiation of the Fibre")

    @property
    def path_lengths(self) -> np.ndarray:
        """
        [np.ndarray] array of fibre path segment lengths in km, shape [S, 2] where S is the number of path segments
        """
        if self._path_coordinates is not None:
            return self._path_lengths
        raise AttributeError("Path coordinates were not passed to Fibre constructor, and path lengths are therefore not available")

    @path_lengths.setter
    def path_lengths(self, value):
        raise AttributeError("The path lengths path_lengths cannot be changed after instantiation of the Fibre")

    @property
    def path_positions(self) -> np.ndarray:
        """
        [np.ndarray] array of accumulative fibre path segment lengths in km, shape [S+1, 2] where S is the number of path segments
        """
        if self._path_positions is not None:
            return self._path_positions
        raise AttributeError("Path coordinates were not passed to Fibre constructor, and path positions are therefore not available")

    @path_positions.setter
    def path_positions(self, value):
        raise AttributeError("The path positions path_positions cannot be changed after instantiation of the Fibre")

    @property
    def section_coordinates(self) -> np.ndarray:
        """
        [np.ndarray] array of fibre section coordinates, shape [S+1, 2] where S is the number of fibre correlation lengths and the second dimension contains longitude, latitude
        """
        if self._section_coordinates is not None:
            return self._section_coordinates
        raise AttributeError("Path coordinates were not passed to Fibre constructor, and section coordinates are therefore not available")

    @section_coordinates.setter
    def section_coordinates(self, value):
        raise AttributeError("The section coordinates section_coordinates cannot be changed after instantiation of the Fibre")

    @property
    def section_lengths(self) -> np.ndarray:
        """
        [np.ndarray] array of fibre section lengths, shape [S] where S is the number of fibre correlation lengths
        """
        return self._section_lengths

    @section_lengths.setter
    def section_lengths(self, value):
        raise AttributeError("The section lengths section_lengths cannot be changed after instantiation of the Fibre")

    @property
    def section_positions(self) -> np.ndarray:
        """
        [np.ndarray] array of accumulative fibre correlation lengths in km, shape [S+1] where S is the number of fibre correlation lengths
        """
        return self._section_positions

    @section_positions.setter
    def section_positions(self, value):
        raise AttributeError("The section positions section_positions cannot be changed after instantiation of the Fibre")

    @property
    def section_count(self) -> int:
        """
        [int] number of fibre sections of this Fibre.
        """
        return len(self.section_lengths)

    @section_count.setter
    def section_count(self, value):
        raise AttributeError("The section count section_count cannot be changed after instantiation of the Fibre.")

    @property
    def PMD_parameter(self) -> float:
        """
        [float] polarisation mode dispersion parameter of this Fibre in ps/(km ^ 0.5)
        """
        return float(self._PMD_parameter)

    @PMD_parameter.setter
    def PMD_parameter(self, value):
        raise AttributeError("The polarisation mode dispersion parameter PMD_parameter cannot be changed after instantiation of the Fibre")

    @property
    def realisation_count(self) -> float:
        """
        [int] realisation count of this Fibre
        """
        return int(self._realisation_count)

    @realisation_count.setter
    def realisation_count(self, value):
        raise AttributeError("The realisation count realisation_count cannot be changed after instantiation of the Fibre")

    @property
    def length(self) -> float:
        """
        [float] total length of the fibre in km
        """
        return self.section_positions[-1]

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
        raise AttributeError("The photoelasticity constant cannot be changed after instantiation of the Fibre")

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
