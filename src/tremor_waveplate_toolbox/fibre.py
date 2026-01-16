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

from .constants import Device, Domain, ModulusModel
from .signal import Signal
from .perturbation import Perturbation
from .path import Path

class Fibre(ABC):
    """
    A base class representing an optical fibre.
    Currently it models only polarisation-mode dispersion using one of two methods:
    - the coarse-step method, which applies differential group delay and scrambles the state of polarisation in a random and distributed manner.
    - Marcuse's method, which was derived directly from the coupled nonlinear Schrödinger equation.
    Chromatic dispersion, the Kerr effect, attenuation, EDFA noise, and polarisation-dependent loss are neglected, and slow polarisation mode dispersion drift are not implemented.
    External perturbations can be modelled as a change in differential phase or major birefringence axes orientations.
    """
    def __init__(self, parameters: ConfigParser):
        """
        Instantiate multiple fibre channels for simultaneous propagation.

        Required entries in parameters['FIBRE']:
        - correlation_length [float]:           correlation length in km
        - beat_length [float]:                  beat length in km
        - section_length [float]:               fibre section length in km (at least the correlation length Lc for the coarse-step model, << Lc for Marcuse's model)
        - path_coordinates [list]:              list of coordinates (longitude, latitude) along the fibre path.
        - chromatic_dispersion [float]:         chromatic dispersion parameter in ps^2/km
        - nonlinearity [float]:                 nonlinearity parameter in  1/(W km)
        - polarisation_mode_dispersion [float]: average accumulated differential group delay in ps/(km ^ 0.5). If 0, turns off major axes rotations as well.
        - realisation_count [int]:              the number of fibre realisations with different distributed polarisation mode dispersion.
        - photoelasticity [float]:              the fibre photoelasticity.
        """
        assert 'FIBRE'                        in parameters, f"Parameters are missing section 'FIBRE'."
        assert 'correlation_length'           in parameters['FIBRE'], f"'correlation_length' is missing from parameters section 'FIBRE'."
        assert 'beat_length'                  in parameters['FIBRE'], f"'beat_length' is missing from parameters section 'FIBRE'."
        assert 'section_length'               in parameters['FIBRE'], f"'section_length' is missing from parameters section 'FIBRE'."
        assert 'chromatic_dispersion'         in parameters['FIBRE'], f"'chromatic_dispersion' is missing from parameters section 'FIBRE'"
        assert 'nonlinearity'                 in parameters['FIBRE'], f"'nonlinearity' is missing from parameters section 'FIBRE'"
        assert 'polarisation_mode_dispersion' in parameters['FIBRE'], f"'polarisation_mode_dispersion' is missing from parameters section 'FIBRE'."
        assert 'realisation_count'            in parameters['FIBRE'], f"'realisation_count' is missing from parameters section 'FIBRE'."
        assert 'photoelasticity'              in parameters['FIBRE'], f"'photoelasticity' is missing from parameters section 'FIBRE'"
        assert 'path_coordinates'             in parameters['FIBRE'] or 'section_count' in parameters['FIBRE'], f"Parameters section 'FIBRE' must contain variable 'path_coordinates' or 'section_count'."
        assert 'modulus_model'                in parameters['FIBRE'], f"'modulus_model' is missing from parameters section 'FIBRE'."

        self._correlation_length           = parameters.getfloat('FIBRE', 'correlation_length')
        self._beat_length                  = parameters.getfloat('FIBRE', 'beat_length')
        self._section_length               = parameters.getfloat('FIBRE', 'section_length')
        self._chromatic_dispersion         = parameters.getfloat('FIBRE', 'chromatic_dispersion')
        self._nonlinearity                 = parameters.getfloat('FIBRE', 'nonlinearity')
        self._polarisation_mode_dispersion = parameters.getfloat('FIBRE', 'polarisation_mode_dispersion')
        self._realisation_count            = int(parameters.getfloat('FIBRE', 'realisation_count'))
        self._photoelasticity              = parameters.getfloat('FIBRE', 'photoelasticity')
        self._modulus_model                = ModulusModel[parameters.get('FIBRE', 'modulus_model')]
        self._material                     = refractiveindex.RefractiveIndexMaterial(
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
        self._init_birefringence()

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

    def _init_birefringence(self):
        """
        Generate differential phase shifts, group delays, and major axes orientations or scramblers.
        """
        match self.modulus_model:
            case ModulusModel.FIXED:  self._init_birefringence_fixed()
            case ModulusModel.RANDOM: self._init_birefringence_random()
            case _: raise AssertionError(f"modulus_model must be ModulusModel.FIXED or ModulusModel.RANDOM, but was {self.modulus_model}")

    @abstractmethod
    def _init_birefringence_fixed(self):
        """
        Initialise birefringences with equal moduli, e.g. using the fixed modulus model.
        """

    @abstractmethod
    def _init_birefringence_random(self):
        """
        Initialise birefringences with random moduli, e.g. using the random modulus model.
        """

    def __call__(self, signal: Signal, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = []) -> Signal:
        """
        Make fibre instances callable; see propagate()
        """
        return self.propagate(signal, transmission_start_times, perturbations)

    def propagate(self, signal: Signal, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = []) -> Signal:
        """
        Propagate a polarisation-multiplexed phase-multiplexed signal through the fibre.
        Multiple fibre realisations are applied at once.
        The calculations will be done on the device (CPU or GPU) that signal resides in (see Signal.to_device())

        Inputs:
        - signal [Signal]: the signal to propagate through the channel, shape [R,B,S,P] with number of realisations R or R = 1, batch size T, sample count S and principal polarisations P = 2.
        - transmission_start_times [float, np.ndarray]: timestamp(s) at which the signal transmission(s) begins in s, relative to the start time of a perturbation. If not a float, shape [T,].
        - perturbations [Perturbation, list]: Model these perturbations during signal transmission, in order of appearance. All birefringence scaling is applied before addition.

        Outputs:
        - [Signal]: the output signal, shape [R,B,S,P] or [R,T,S,P]
        """
        assert signal.sample_axis_negative == -2, f"signal must have sample axis -2, but it was {signal.sample_axis_negative}"
        assert signal.shape[-1] == 2, f"signal must have two polarisations on the last axis, but had {signal.shape[-1]}"
        assert len(signal.shape) == 4, f"signal must have shape [R,B,S,P], but had shape {signal.shape}"
        if signal.device == Device.CUDA: signal.to_device(Device.CUDA) # Ensure that the signal resides in the currently active cupy GPU
        
        signal = self._propagate_master(
                signal.copy(),
                signal.frequency_angular,
                transmission_start_times,
                perturbations
            )

        return signal

    def Jones(self, frequency_angular: (np.ndarray), carrier_wavelength: float = 1550., transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = []) -> np.ndarray:
        """
        Calculate the fibre Jones matrix.
        The calculations will be done on the device (CPU or GPU) that frequency_angular resides in (see Signal.to_device())
        The matrix will exclude any noise or nonlinear effects.

        Inputs:
        - frequency_angular [np.ndarray, cp.ndarray]: frequencies in rad/s at which to calculate the jones matrix, relative to the carrier frequency, shape [S,]
        - carrier_wavelength [float]: carrier wavelength in nm
        - transmission_start_times [float, np.ndarray]: timestamp(s) at which the signal transmission(s) begins in s, relative to the start time of a perturbation. If not a float, shape [T,].
        - perturbations [Perturbation, list]: Model these perturbations during signal transmission, in order of appearance. All birefringence scaling is applied before addition.

        Outputs:
        - [np.ndarray, cp.ndarray]: the Jones matrices, shape [R,T,S,2,2]
        """
        if 'cupy' in sys.modules and isinstance(frequency_angular, cp.ndarray):
            xp = cp
            frequency_angular = xp.array(frequency_angular) # Ensure that the frequency array resides in the currently active cupy GPU
        else:
            xp = np

        assert len(frequency_angular.shape) == 1, f"frequency_angular must have shape [F,], but had shape {frequency_angular.shape}"
        assert self.nonlinearity == 0, f"Jones matrices can only be calculated for fibres with a nonlinearity coefficient of 0."
        
        Jones_matrix_transposed = Signal(
                samples = xp.eye(2, dtype = complex)[None, None, None],
                sample_rate = 1, # Placeholder value
                sample_axis = -3,
                domain = Domain.FREQUENCY,
                carrier_wavelength = carrier_wavelength
            )

        Jones_matrix_transposed = self._propagate_master(
                Jones_matrix_transposed,
                frequency_angular,
                transmission_start_times,
                perturbations
            )

        return xp.moveaxis(Jones_matrix_transposed.samples, -1, -2)

    def accumulate_differential_group_delays(self, device: Device = Device.CPU) -> np.ndarray:
        """
        Accumulated differential group delay in ps, shape [R] where R is the number of fibre realisations
        """
        match(device):
            case Device.CPU: xp = np
            case Device.CUDA:
                assert 'cupy' in sys.modules, f"Cannot calculate differential group delay on CUDA; no cupy installation found"
                xp = cp
                
        frequency_angular = xp.array([-xp.pi, xp.pi]) / (60 * self.section_path.lengths[0])

        Jones_matrices = self.Jones(frequency_angular)[:, 0] # [R,F,2,2]
        Jones_matrices_derivative = xp.diff(Jones_matrices, axis = 1)[:, 0] / xp.diff(frequency_angular)[0]

        differential_group_delay = 2 * xp.sqrt(xp.linalg.det(Jones_matrices_derivative)) * 1e12 # Gordon et al. - PMD Fundamentals: Polarization Mode Dispersion in Optical Fibres
        differential_group_delay = differential_group_delay.real.astype(float)

        if xp != np: differential_group_delay = differential_group_delay.get()

        return differential_group_delay

    @abstractmethod
    def _propagate_master(self, signal: Signal, frequency_angular: np.ndarray, transmission_start_times: (float, np.ndarray) = 0, perturbations: (Perturbation, list) = []) -> Signal:
        """
        Method called by propagate() and Jones() to simulate the fibre response.

        Inputs:
        - signal [Signal]: the signal or transposed Jones matrix to propagate through the channel, shape [R,B,S,P] with number of realisations R or R = 1, batch size B, sample count S and principal polarisations P = 2, OR shape [R, S, P, P] where the last two axes contain Jones transfer matrices.
        - frequency_angular [np.ndarray, cp.ndarray]: frequencies of the signal or Jones matrix in rad/s, relative to the carrier frequency, shape [S,]
        - transmission_start_times [float, np.ndarray]: timestamp(s) at which the signal transmission(s) begins in s, relative to the start time of a perturbation. If not a float, shape [T,].
        - perturbations [Perturbation, list]: Model these perturbations during signal transmission, in order of appearance. All birefringence scaling is applied before addition.

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
            'correlation_length':           self.correlation_length,
            'beat_length':                  self.beat_length,
            'section_path':                 self.section_path.to_dict(),
            'polarisation_mode_dispersion': self.polarisation_mode_dispersion,
            'chromatic_dispersion':         self.chromatic_dispersion,
            'nonlinearity':                 self.nonlinearity,
            'realisation_count':            self.realisation_count,
            'photoelasticity':              self.photoelasticity,
            'modulus_model':                self.modulus_model.name,
            'differential_group_delays':    self.differential_group_delays.tolist()
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
        parameters.set('FIBRE', 'correlation_length',           str(fibre_dict['correlation_length']))
        parameters.set('FIBRE', 'beat_length',                  str(fibre_dict['beat_length']))
        parameters.set('FIBRE', 'chromatic_dispersion',         str(fibre_dict['chromatic_dispersion']))
        parameters.set('FIBRE', 'nonlinearity',                 str(fibre_dict['nonlinearity']))
        parameters.set('FIBRE', 'polarisation_mode_dispersion', str(fibre_dict['polarisation_mode_dispersion']))
        parameters.set('FIBRE', 'realisation_count',            str(fibre_dict['realisation_count']))
        parameters.set('FIBRE', 'photoelasticity',              str(fibre_dict['photoelasticity']))
        parameters.set('FIBRE', 'modulus_model',                fibre_dict['modulus_model'])

        section_path = Path.from_dict(fibre_dict['section_path'])
        parameters.set('FIBRE', 'section_length', str(section_path.lengths[0]))
        
        if 'path' in fibre_dict:
            path = Path.from_dict(fibre_dict['path'])
            parameters.set('FIBRE', 'path_coordinates', json.dumps(path.coordinates.tolist()))
        else:
            parameters.set('FIBRE', 'section_count', str(section_path.edge_count))

        fibre = cls(parameters)
        fibre.differential_group_delays = np.array(fibre_dict['differential_group_delays'])
        
        fibre._section_path = section_path
        if 'path' in fibre_dict:
            fibre._path = path

        return fibre

    def __eq__(self, other) -> bool:
        return self._polarisation_mode_dispersion  == other._polarisation_mode_dispersion and \
            self._photoelasticity                  == other._photoelasticity              and \
            self._realisation_count                == other._realisation_count            and \
            self._section_path                     == other._section_path                 and \
            self._path                             == other._path                         and \
            np.all(self._differential_group_delays == other._differential_group_delays)   and \
            self._modulus_model                    == other._modulus_model

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
    def chromatic_dispersion(self) -> float:
        """
        [float] chromatic dispersion parameter of this Fibre in ps ^ 2 / km / rad
        """
        return float(self._chromatic_dispersion)

    @chromatic_dispersion.setter
    def chromatic_dispersion(self, value):
        raise AttributeError("The chromatic dispersion parameter chromatic_dispersion cannot be changed after instantiation of the Fibre")

    @property
    def nonlinearity(self) -> float:
        """
        [float] nonlinearity parameter of this Fibre in rad / (W km)
        """
        return float(self._nonlinearity)

    @nonlinearity.setter
    def nonlinearity(self, value):
        raise AttributeError("The nonlinearity parameter cannot be changed after instantiation of the Fibre")

    @property
    def polarisation_mode_dispersion(self) -> float:
        """
        [float] polarisation mode dispersion parameter of this Fibre in ps / (km ^ 0.5)
        """
        return float(self._polarisation_mode_dispersion)

    @polarisation_mode_dispersion.setter
    def polarisation_mode_dispersion(self, value):
        raise AttributeError("The polarisation mode dispersion parameter polarisation_mode_dispersion cannot be changed after instantiation of the Fibre")

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
    def differential_group_delays(self) -> np.ndarray:
        """
        [float] the differential group delay per section and realisation in ps. Shape [S, R] where S is the number of fibre sections and R the number of realisations.
        """
        return self._differential_group_delays

    @differential_group_delays.setter
    def differential_group_delays(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), f"New differential_group_delays must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New differential_group_delays array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.section_path.edge_count, self.realisation_count), f"New differential_group_delays array must have shape (self.section_path.edge_count ({self.section_path.edge_count}), self.realisation_count ({self.realisation_count})), but had shape {value.shape}"
        self._differential_group_delays = value.copy().astype(float)

    @property
    def differential_group_delay(self) -> np.ndarray:
        """
        Accumulated differential group delay in ps, shape [R] where R is the number of fibre realisations
        """
        return self.accumulate_differential_group_delays()

    @differential_group_delay.setter
    def differential_group_delay(self, value):
        raise AttributeError("The accumulated differential_group_delay cannot be set directly; set section differential_group_delays instead")

    @property
    def modulus_model(self) -> ModulusModel:
        """
        The modulus model with which the birefringence is initialised (FIXED or RANDOM)
        """
        return self._modulus_model

    @modulus_model.setter
    def modulus_model(self, value):
        raise AttributeError("The fibre birefringence modulus model cannot be changed after instantiation")