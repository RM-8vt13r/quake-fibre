"""
An optical fibre channel model base class for dual-polarisation transmission.
"""

from configparser import ConfigParser
import json
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import obspy as op
import refractiveindex

from .signal import Signal

class Fibre(ABC):
    """
    A base class representing an optical fibre.
    Currently it models only polarisation-mode dispersion using one of two methods:
    - the coarse-step method, which applies differential group delay and scrambles the state of polarisation in a random and distributed manner.
    - Marcuse's method, which was derived directly from the coupled nonlinear Schrödinger equation.
    Chromatic dispersion, the Kerr effect, attenuation, EDFA noise, and polarisation-dependent loss are neglected, and slow PMD drift are not implemented.
    Earthquake strain can be modelled as a change in differential group delay.
    """
    def __init__(self, parameters: ConfigParser):
        """
        Instantiate multiple fibre channels for simultaneous propagation.

        Required entries in parameters['FIBRE']:
        - correlation_length [float]: correlation length in km
        - beat_length [float]:        beat length in km
        - section_length [float]:     PMD section length in km (at least the correlation length Lc for the coarse-step model, << Lc for Marcuse's model)
        - path_coordinates [list]:    list of coordinates (longitude, latitude) along the fibre path.
        - PMD_parameter [float]:      average accumulated differential group delay in ps/(km ^ 0.5).
        - realisation_count [int]:    the number of fibre realisations with different distributed polarisation mode dispersion.
        - photoelasticity [float]:    the fibre photoelasticity.
        """
        assert 'FIBRE'              in parameters, f"Parameters are missing section 'FIBRE'."
        assert 'correlation_length' in parameters['FIBRE'], f"'correlation_length' is missing from parameters section 'FIBRE'."
        assert 'beat_length'        in parameters['FIBRE'], f"'beat_length' is missing from parameters section 'FIBRE'."
        assert 'section_length'     in parameters['FIBRE'], f"'section_length' is missing from parameters section 'FIBRE'."
        assert 'PMD_parameter'      in parameters['FIBRE'], f"'PMD_parameter' is missing from parameters section 'FIBRE'."
        assert 'realisation_count'  in parameters['FIBRE'], f"'realisation_count' is missing from parameters section 'FIBRE'."
        assert 'photoelasticity'    in parameters['FIBRE'], f"'photoelasticity' is missing from parameters section 'FIBRE'"
        assert 'path_coordinates'   in parameters['FIBRE'] or 'section_count' in parameters['FIBRE'], f"Parameters section 'FIBRE' must contain variable 'path_coordinates' or 'section_count'."

        self._correlation_length = parameters.getfloat('FIBRE', 'correlation_length')
        self._beat_length        = parameters.getfloat('FIBRE', 'beat_length')
        self._section_length     = parameters.getfloat('FIBRE', 'section_length')
        self._PMD_parameter      = parameters.getfloat('FIBRE', 'PMD_parameter')
        self._realisation_count  = int(parameters.getfloat('FIBRE', 'realisation_count'))
        self._photoelasticity    = parameters.getfloat('FIBRE', 'photoelasticity')
        self._material           = refractiveindex.RefractiveIndexMaterial(
            shelf = 'glass',
            book  = 'fused_silica',
            page  = 'Malitson'
        )
        
        if 'path_coordinates' in parameters['FIBRE']:
            self._path_coordinates = np.array(json.loads(parameters.get('FIBRE', 'path_coordinates')), dtype = float)
            self._section_count    = None
        else:
            self._path_coordinates = None
            self._section_count    = parameters.getint('FIBRE', 'section_count')

        self._init_path()
        self._init_DGD()
        self._init_PSP()

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
            
            section_longitudes        = np.interp(self.section_positions, self.path_positions, self.path_coordinates[:, 0])
            section_latitudes         = np.interp(self.section_positions, self.path_positions, self.path_coordinates[:, 1])
            self._section_coordinates = np.stack([section_longitudes, section_latitudes], axis = 1)

            self._section_lengths = np.diff(self.section_positions)

        else:
            self._path_lengths           = None
            self._path_positions         = None
            self._section_coordinates    = None
            self._section_lengths        = np.full(
                shape      = (self._section_count,),
                fill_value = self._section_length,
                dtype      = float
            )

            self._section_positions      = np.append([0], np.cumsum(self.section_lengths))
            
    @abstractmethod
    def _init_DGD(self):
        """
        Initialise random differential group delay per realisation and fibre section in ps.
        """
        pass

    @abstractmethod
    def _init_PSP(self):
        """
        Initialise a random preferred orientation and/or scrambling of the state of polarisation per realisation and fibre section
        """
        pass

    def __call__(self, signal: Signal, strain: np.ndarray = None, transmission_start_time: float = 0, verbose: bool = False) -> Signal:
        """
        Make fibre instances callable; see propagate()
        """
        return self.propagate(signal, strain, transmission_start_time, verbose)

    @abstractmethod
    def propagate(self, signal: Signal, strain: Signal = None, transmission_start_time: float = 0, verbose: bool = False) -> Signal:
        """
        Propagate a polarisation-multiplexed phase-multiplexed signal through the fibre.
        Multiple fibre realisations are applied at once.
        The calculations will be done on the device (CPU or GPU) that signal resides in (see Signal.to_device())

        Inputs:
        - signal [Signal]: the signal to propagate through the channel in the time domain, shape [R,B,S,P] with number of realisations R or R = 1, batch size B, sample count S and principal polarisations P = 2.
        - strain [Signal]: If not None, contains longitudinal strain values per fibre section over time; shape [F,T] with fibre sections F and strain samples T
        - transmission_start_time [float]: timestamp at which the signal transmission begins in s
        - verbose [bool]: whether to show a progress bar

        Outputs:
        - [Signal]: the output signal, shape [R,B,S,P]
        """
        pass

    @abstractmethod
    def Jones(self, frequency_angular: (np.ndarray), strain: Signal = None, transmission_start_time: float = 0, verbose: bool = False) -> np.ndarray:
        """
        Calculate the fibre Jones matrix in the absence of external perturbations.
        The calculations will be done on the device (CPU or GPU) that signal resides in (see Signal.to_device())

        Inputs:
        - frequency_angular [np.ndarray, cp.ndarray]: frequencies in rad/s at which to calculate the jones matrix, relative to the carrier frequency, shape [F,]
        - strain [Signal]: If not None, contains longitudinal strain values per fibre section over time; shape [F,T] with fibre sections F and strain samples T
        - transmission_start_time [float]: timestamp at which the signal transmission begins in s
        - verbose [bool]: whether to show a progress bar

        Outputs:
        - [np.ndarray, cp.ndarray]: the Jones matrices, shape [R,F,2,2]
        """
        pass

    def group_velocity(self, signal: Signal):
        """
        Obtain the fibre propagation constant (inverse of group velocity) for a specific signal in km/s

        Inputs:
        - signal [Signal]: the signal for which to obtain the propagation constant

        Outputs:
        - [float] group velocity in km/s
        """
        return sp.constants.speed_of_light / self.material.get_refractive_index(signal.carrier_wavelength) / 1000

    def to_dict(self) -> dict:
        """
        Represent this fibre (and its exact realisations) as a dictionary.

        Outputs:
        - [dict] The dictionary representation of this fibre
        """
        fibre_dict = {
            'correlation_length':         self.correlation_length,
            'beat_length':                self.beat_length,
            'section_lengths':            self.section_lengths.tolist(),
            'section_positions':          self.section_positions.tolist(),
            'PMD_parameter':              self.PMD_parameter,
            'realisation_count':          self.realisation_count,
            'photoelasticity':            self.photoelasticity,
            'section_DGD':                self.section_DGD.tolist()
        }

        if self._section_major_angles is not None:
            fibre_dict = fibre_dict | {
                'section_major_angles':   self.section_major_angles.tolist(),
                'section_birefringences': self.section_birefringences.tolist()
            }

        else:
            fibre_dict = fibre_dict | {
                'section_PSP': self.section_PSP.tolist()
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
        parameters.set('FIBRE', 'correlation_length', str(fibre_dict['correlation_length']))
        parameters.set('FIBRE', 'beat_length',        str(fibre_dict['beat_length']))
        parameters.set('FIBRE', 'section_length',     str(fibre_dict['section_lengths'][0]))
        parameters.set('FIBRE', 'PMD_parameter',      str(fibre_dict['PMD_parameter']))
        parameters.set('FIBRE', 'realisation_count',  str(fibre_dict['realisation_count']))
        parameters.set('FIBRE', 'photoelasticity',    str(fibre_dict['photoelasticity']))

        if 'path_coordinates' in fibre_dict:
            parameters.set('FIBRE', 'path_coordinates', str(fibre_dict['path_coordinates']))
        else:
            parameters.set('FIBRE', 'section_count', str(len(fibre_dict['section_lengths'])))

        fibre = cls(parameters)
        fibre.section_DGD                = np.array(fibre_dict['section_DGD'])

        if 'section_major_angles' in fibre_dict:
            fibre.section_major_angles   = np.array(fibre_dict['section_major_angles'])
            fibre.section_birefringences = np.array(fibre_dict['section_birefringences'])
        else:
            fibre.section_PSP            = np.array(fibre_dict['section_PSP'])

        return fibre

    def __eq__(self, other):
        if self._PMD_parameter                  == other._PMD_parameter        and \
            self._photoelasticity               == other._photoelasticity      and \
            self._realisation_count             == other._realisation_count    and \
            self._section_count                 == other._section_count        and \
            self._section_length                == other._section_length       and \
            np.all(self._path_coordinates       == other._path_coordinates)    and \
            np.all(self._path_lengths           == other._path_lengths)        and \
            np.all(self._path_positions         == other._path_positions)      and \
            np.all(self._section_coordinates    == other._section_coordinates) and \
            np.all(self._section_lengths        == other._section_lengths)     and \
            np.all(self._section_positions      == other._section_positions)   and \
            np.all(self._section_DGD            == other._section_DGD)         and \
            np.all(self._section_major_angles   == other._section_major_angles)  and \
            np.all(self._section_birefringences == other._section_birefringences)  and \
            np.all(self._section_PSP            == other._section_PSP):
            return True

        return False

    @property
    def path_coordinates(self) -> np.ndarray:
        """
        [np.ndarray] array of fibre path segment endpoint coordinates, shape [S+1, 2] where S is the number of path segments and the second dimension contains longitude, latitude
        """
        if self._path_coordinates is not None:
            return self._path_coordinates
        raise AttributeError("Path coordinates were not passed to Fibre constructor, and are therefore not available")

    @path_coordinates.setter
    def path_coordinates(self, value):
        raise AttributeError("The path endpoint coordinates path_coordinates cannot be changed after instantiation of the Fibre")

    @property
    def path_centre_coordinates(self) -> np.ndarray:
        """
        [np.ndarray] array of fibre path segment midpoint coordinates, shape [S, 2]
        """
        return (self.path_coordinates[:-1] + self.path_coordinates[1:]) / 2

    @path_centre_coordinates.setter
    def path_centre_coordinates(self, value):
        raise AttributeError("The path midpoint coordinates cannot be set directly")

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
        raise AttributeError("The path segment endpoint positions path_positions cannot be changed after instantiation of the Fibre")

    @property
    def path_centre_positions(self):
        """
        [np.ndarray] array of accumulative fibre path segment lengths in km measured at each segment centre, shape [S, 2] where S is the number of path segments
        """
        return (self.path_positions[:-1] + self.path_positions[1:]) / 2

    @path_centre_positions.setter
    def path_centre_positions(self, value):
        raise AttributeError("The path centre positions path_positions cannot be changed after instantiation of the Fibre")

    @property
    def section_coordinates(self) -> np.ndarray:
        """
        [np.ndarray] array of fibre section coordinates, shape [S+1, 2] where S is the number of fibre sections and the second dimension contains longitude, latitude
        """
        if self._section_coordinates is not None:
            return self._section_coordinates
        raise AttributeError("Path coordinates were not passed to Fibre constructor, and section coordinates are therefore not available")

    @section_coordinates.setter
    def section_coordinates(self, value):
        raise AttributeError("The section coordinates section_coordinates cannot be changed after instantiation of the Fibre")

    @property
    def section_centre_coordinates(self) -> np.ndarray:
        """
        [np.ndarray] array of fibre section midpoint coordinates, shape [S, 2]
        """
        return (self.section_coordinates[:-1] + self.section_coordinates[1:]) / 2

    @section_centre_coordinates.setter
    def section_centre_coordinates(self, value):
        raise AttributeError("The section midpoint coordinates cannot be set directly")

    @property
    def section_lengths(self) -> np.ndarray:
        """
        [np.ndarray] array of fibre section lengths, shape [S] where S is the number of fibre sections
        """
        return self._section_lengths

    @section_lengths.setter
    def section_lengths(self, value):
        raise AttributeError("The section lengths section_lengths cannot be changed after instantiation of the Fibre")

    @property
    def section_positions(self) -> np.ndarray:
        """
        [np.ndarray] array of accumulative fibre sections in km, shape [S+1] where S is the number of fibre sections
        """
        return self._section_positions

    @section_positions.setter
    def section_positions(self, value):
        raise AttributeError("The section endpoint positions section_positions cannot be changed after instantiation of the Fibre")

    @property
    def section_centre_positions(self) -> np.ndarray:
        """
        [np.ndarray] array of accumulative fibre sections in km measured at each section centre, shape [S, 2] where S is the number of fibre sections
        """
        return (self.section_positions[:-1] + self.section_positions[1:]) / 2

    @section_centre_positions.setter
    def section_centre_positions(self, value):
        raise AttributeError("The section centre positions section_positions cannot be changed after instantiation of the Fibre")

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
    def correlation_length(self) -> float:
        """
        [float] the fibre correlation length in km, after which the state of polarisation is uncorrelated to its initial state
        """
        return self._correlation_length

    @correlation_length.setter
    def correlation_length(self, value):
        raise AttributeError("The correlation length correlation_length cannot be changed after instantiation of the Fibre")

    @property
    def beat_length(self) -> float:
        """
        [float] the fibre beat length in km, in which the differential phase rotates 2pi radians
        """
        return self._beat_length

    @beat_length.setter
    def beat_length(self, value):
        raise AttributeError("The beat length beat_length cannot be changed after instantiation of the Fibre")

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
    def material(self):
        """
        [RefractiveIndexMaterial] The fibre material, used to determine group velocity from carrier wavelength.
        """
        return self._material

    @material.setter
    def material(self, value):
        raise AttributeError("The fibre material cannot be changed after instantiation")

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
        Accumulated differential group delay in ps, shape [R] where R is the number of fibre realisations
        """
        frequency_angular = np.array([-np.pi, np.pi]) / (60 * self.section_lengths[0])

        Jones_matrices = self.Jones(frequency_angular) # [R,F,2,2]
        Jones_matrices_derivative = np.diff(Jones_matrices, axis = 1)[:, 0] / np.diff(frequency_angular)[0]

        accumulated_DGD = 2 * np.sqrt(np.linalg.det(Jones_matrices_derivative)) * 1e12 # Gordon et al. - PMD Fundamentals: Polarization Mode Dispersion in Optical Fibres
        accumulated_DGD = accumulated_DGD.real.astype(float)

        return accumulated_DGD

    @DGD.setter
    def DGD(self, value):
        raise AttributeError("The accumulated DGD cannot be set directly; set section DGD instead")

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
        assert value.shape == (self.section_count, self.realisation_count), f"New section_major_angles array must have shape (self.section_count ({self.section_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
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
        assert value.shape == (self.section_count, self.realisation_count), f"New section_birefringences array must have shape (self.section_count ({self.section_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        # assert np.all((value >= -np.pi) & (value < np.pi)), f"New section_birefringences array must have values between -pi and pi"
        self._section_birefringences = value.copy().astype(float)

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
        assert value.shape == (self.section_count, self.realisation_count, 2, 2), f"New section_PSP array must have shape (self.section_count ({self.section_count}), self.realisation_count ({self.realisation_count}), 2, 2), but had shape {value.shape}"
        assert np.allclose(value[..., 0, 0], value[..., 1, 1].conjugate()) and np.allclose(value[..., 1, 0], -value[..., 0, 1].conjugate()), f"New section_PSP array must contain unitary matrices, but didn't"
        self._section_PSP = value.copy().astype(complex)