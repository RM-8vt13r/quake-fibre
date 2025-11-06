"""
An optical fibre channel model base class for dual-polarisation transmission.
"""

from configparser import ConfigParser
import json
import sys
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
try:
    import cupy as cp
except:
    pass
import obspy as op
import refractiveindex

from .constants import Device, Domain
from .earthquake import Earthquake
from .signal import Signal
from .path import Path

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
            self._path = Path(
                    *np.array(json.loads(parameters.get('FIBRE', 'path_coordinates')), dtype = float).T
                )
            self._section_count    = None
        else:
            self._path          = None
            self._section_count = parameters.getint('FIBRE', 'section_count')

        self._init_path()
        self._init_DGD()
        self._init_PSP()

    def _init_path(self):
        """
        Initialise fibre- and section path information.
        """
        if self._path is not None:
            section_positions  = np.append(np.arange(0, self.path.positions[-1], self._section_length), self.path.positions[-1])
            section_longitudes = np.interp(section_positions, self.path.positions, self.path.longitudes)
            section_latitudes  = np.interp(section_positions, self.path.positions, self.path.latitudes)
            self._section_path  = Path(section_longitudes, section_latitudes)

        else:
            self._section_path = Path(lengths = np.full(
                    shape      = (self._section_count,),
                    fill_value = self._section_length,
                    dtype      = float
                ))

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

    def __call__(self, signal: Signal, transmission_start_time: float = 0, earthquake: Earthquake = None, earthquake_batch_size: int = 100, verbose: bool = False) -> Signal:
        """
        Make fibre instances callable; see propagate()
        """
        return self.propagate(signal, transmission_start_time, earthquake, earthquake_batch_size, verbose)

    def propagate(self, signal: Signal, transmission_start_time: float = 0, earthquake: Earthquake = None, earthquake_batch_size: int = 100, verbose: bool = False) -> Signal:
        """
        Propagate a polarisation-multiplexed phase-multiplexed signal through the fibre.
        Multiple fibre realisations are applied at once.
        The calculations will be done on the device (CPU or GPU) that signal resides in (see Signal.to_device())

        Inputs:
        - signal [Signal]: the signal to propagate through the channel, shape [R,B,S,P] with number of realisations R or R = 1, batch size B, sample count S and principal polarisations P = 2.
        - transmission_start_time [float]: timestamp at which the signal transmission begins in s, relative to the start time of a potential earthquake
        - earthquake [Earthquake]: If not None, model the strain of this earthquake during signal transmission
        - earthquake_batch_size [int]: how many seismograms to request from Syngine at a time; should always be below 5000
        - verbose [bool]: whether to show a progress bar

        Outputs:
        - [Signal]: the output signal, shape [R,B,S,P]
        """
        assert signal.sample_axis_negative == -2, f"signal must have sample axis -2, but it was {signal.sample_axis_negative}"
        assert signal.shape[-1] == 2, f"signal must have two polarisations on the last axis, but had {signal.shape[-1]}"
        assert len(signal.shape) == 4, f"signal must have shape [R,B,S,P], but had shape {signal.shape}"
        if signal.device == Device.CUDA: signal.to_device(Device.CUDA) # Ensure that the signal resides in the currently active cupy GPU
        
        signal = self._propagate_master(
                signal,
                signal.frequency_angular,
                transmission_start_time,
                earthquake,
                earthquake_batch_size,
                verbose
            )

        return signal

    def Jones(self, frequency_angular: (np.ndarray), carrier_wavelength: float = 1550., transmission_start_time: float = 0, earthquake: Earthquake = None, earthquake_batch_size = 100, verbose: bool = False) -> np.ndarray:
        """
        Calculate the fibre Jones matrix.
        The calculations will be done on the device (CPU or GPU) that frequency_angular resides in (see Signal.to_device())
        The matrix will exclude any noise or nonlinear effects.

        Inputs:
        - frequency_angular [np.ndarray, cp.ndarray]: frequencies in rad/s at which to calculate the jones matrix, relative to the carrier frequency, shape [F,]
        - carrier_wavelength [float]: carrier wavelength in nm
        - transmission_start_time [float]: timestamp at which the signal transmission begins in s, relative to the start time of a potential earthquake
        - earthquake [Earthquake]: If not None, hold on to your hats!
        - earthquake_batch_size [int]: how many seismograms to request from Syngine at a time; should always be below 5000
        - verbose [bool]: whether to print progress bars and messages

        Outputs:
        - [np.ndarray, cp.ndarray]: the Jones matrices, shape [R,F,2,2]
        """
        if 'cupy' in sys.modules and isinstance(frequency_angular, cp.ndarray):
            xp = cp
            frequency_angular = xp.array(frequency_angular) # Ensure that the frequency array resides in the currently active cupy GPU
        else:
            xp = np

        assert len(frequency_angular.shape) == 1, f"frequency_angular must have shape [F,], but had shape {frequency_angular.shape}"
        
        Jones_matrix_transposed = Signal(
                samples = xp.eye(2, dtype = complex)[None, None],
                sample_rate = 1, # Placeholder value
                sample_axis = -3,
                domain = Domain.FREQUENCY,
                carrier_wavelength = carrier_wavelength
            )

        Jones_matrix_transposed = self._propagate_master(
                Jones_matrix_transposed,
                frequency_angular,
                transmission_start_time,
                earthquake,
                earthquake_batch_size,
                verbose
            )

        return xp.transpose(Jones_matrix_transposed.samples, (0, 1, 3, 2))

    @abstractmethod
    def _propagate_master(self, signal: Signal, frequency_angular: np.ndarray, transmission_start_time: float = 0, earthquake: Earthquake = None, earthquake_batch_size: int = 100, verbose: bool = False) -> Signal:
        """
        Method called by propagate() and Jones() to simulate the fibre response.

        Inputs:
        - signal [Signal]: the signal or transposed Jones matrix to propagate through the channel, shape [R,B,S,P] with number of realisations R or R = 1, batch size B, sample count S and principal polarisations P = 2, OR shape [R, S, P, P] where the last two axes contain Jones transfer matrices.
        - frequency_angular [np.ndarray, cp.ndarray]: frequencies of the signal or Jones matrix in rad/s, relative to the carrier frequency, shape [S,]
        - transmission_start_time [float]: timestamp at which the signal transmission begins in s, relative to the start time of a potential earthquake
        - earthquake [Earthquake]: If not None, hold on to your hats!
        - earthquake_batch_size [int]: how many seismograms to request from Syngine at a time; should always be below 5000
        - verbose [bool]: whether to print progress bars and messages

        Outputs:
        - [Signal]: the output signal or transposed Jones matrix, same shape as signal
        """
        pass

    def group_velocity(self, carrier_wavelength: float):
        """
        Obtain the fibre propagation constant (inverse of group velocity) for a specific signal in km/s

        Inputs:
        - carrier_wavelength [float]: the carrier wavelength at which to obtain the propagation constant

        Outputs:
        - [float] group velocity in km/s
        """
        return sp.constants.speed_of_light / self.material.get_refractive_index(carrier_wavelength) / 1000

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Represent this fibre (and its exact realisations) as a dictionary.

        Outputs:
        - [dict] The dictionary representation of this fibre
        """
        fibre_dict = {
            'correlation_length': self.correlation_length,
            'beat_length':        self.beat_length,
            'section_path':       self.section_path.to_dict(),
            'PMD_parameter':      self.PMD_parameter,
            'realisation_count':  self.realisation_count,
            'photoelasticity':    self.photoelasticity,
            'section_DGD':        self.section_DGD.tolist()
        }

        if self._path is not None:
            fibre_dict = fibre_dict | {
                'path': self.path.to_dict()
            }

        return fibre_dict

    @classmethod
    @abstractmethod
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
        parameters.set('FIBRE', 'PMD_parameter',      str(fibre_dict['PMD_parameter']))
        parameters.set('FIBRE', 'realisation_count',  str(fibre_dict['realisation_count']))
        parameters.set('FIBRE', 'photoelasticity',    str(fibre_dict['photoelasticity']))

        section_path = Path.from_dict(fibre_dict['section_path'])
        parameters.set('FIBRE', 'section_length', str(section_path.lengths[0]))
        
        if 'path' in fibre_dict:
            path = Path.from_dict(fibre_dict['path'])
            parameters.set('FIBRE', 'path_coordinates', json.dumps(path.coordinates.tolist()))
        else:
            parameters.set('FIBRE', 'section_count', str(section_path.edge_count))

        fibre = cls(parameters)
        fibre.section_DGD                = np.array(fibre_dict['section_DGD'])
        
        fibre._section_path = section_path
        if 'path' in fibre_dict:
            fibre._path = path

        return fibre

    def __eq__(self, other) -> bool:
        return self._PMD_parameter       == other._PMD_parameter     and \
            self._photoelasticity    == other._photoelasticity   and \
            self._realisation_count  == other._realisation_count and \
            self._section_path       == other._section_path      and \
            self._path               == other._path              and \
            np.all(self._section_DGD == other._section_DGD)

    @property
    def path(self) -> Path:
        """
        [Path] segmented fibre path
        """
        if self._path is None:
            raise AttributeError("Path coordinates were not passed to Fibre constructor, and are therefore not available")
        return self._path

    @path.setter
    def path(self, value):
        raise AttributeError("The path cannot be changed after instantiation of the Fibre")

    @property
    def section_path(self) -> Path:
        """
        [Path] segmented fibre path, divided into short split-step sections. May or may not include earth coordinates, depending on the fibre parameters
        """
        return self._section_path

    @section_path.setter
    def section_path(self, value):
        raise AttributeError("The section path cannot be changed after instantiation of the Fibre")

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
        return self.section_path.positions[-1]

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
        assert value.shape == (self.section_path.edge_count, self.realisation_count), f"New section_DGD array must have shape (self.section_path.edge_count ({self.section_path.edge_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        self._section_DGD = value.copy().astype(float)

    @property
    def DGD(self) -> np.ndarray:
        """
        Accumulated differential group delay in ps, shape [R] where R is the number of fibre realisations
        """
        frequency_angular = np.array([-np.pi, np.pi]) / (60 * self.section_path.lengths[0])

        Jones_matrices = self.Jones(frequency_angular) # [R,F,2,2]
        Jones_matrices_derivative = np.diff(Jones_matrices, axis = 1)[:, 0] / np.diff(frequency_angular)[0]

        accumulated_DGD = 2 * np.sqrt(np.linalg.det(Jones_matrices_derivative)) * 1e12 # Gordon et al. - PMD Fundamentals: Polarization Mode Dispersion in Optical Fibres
        accumulated_DGD = accumulated_DGD.real.astype(float)

        return accumulated_DGD

    @DGD.setter
    def DGD(self, value):
        raise AttributeError("The accumulated DGD cannot be set directly; set section DGD instead")