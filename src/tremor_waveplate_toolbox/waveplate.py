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
        - Lc [float]: correlation length in km.
        - Nsec [int]: the number of fibre sections, each of which has length Lc.
        - tau [float]: average polarisation mode dispersion parameter in ps/(km ^ 0.5).
        - Nreal [int]: the number of polarisation mode dispersion realisations.
        - xi [float]: the fibre photoelasticity.
        """
        super().__init__(parameters)

        assert 'FIBRE' in parameters, f"Parameters are missing section 'FIBRE'."
        assert 'Lc'    in parameters['FIBRE'], f"Correlation length 'Lc' is missing from parameters section 'FIBRE'."
        assert 'Nsec'  in parameters['FIBRE'], f"Section count 'Nsec' is missing from parameters section 'FIBRE'."
        assert 'tau'   in parameters['FIBRE'], f"PMD parameter 'tau' is missing from parameters section 'FIBRE'."
        assert 'Nreal' in parameters['FIBRE'], f"Realisation count 'Nreal' is missing from parameters section 'FIBRE'."
        assert 'xi'    in parameters['FIBRE'], f"Photoelasticity 'xi' is missing from parameters section 'FIBRE'"

        self._Lc    = parameters.getfloat('FIBRE', 'Lc')
        self._Nsec  = parameters.getint('FIBRE', 'Nsec')
        self._tau   = parameters.getfloat('FIBRE', 'tau')
        self._Nreal = parameters.getint('FIBRE', 'Nreal')
        self._xi    = parameters.getfloat('FIBRE', 'xi')

        self.init_DGD()
        self.init_PSP_rotations()
        self.init_material_strain()

    def init_DGD(self):
        """
        Initialise random differential group delay per realisation and fibre section in ps.
        """
        DGD_section_mean  = self.tau * np.sqrt(self.Lc * 3 * np.pi / 8) # Czegledi et al. (2016): Polarization-Mode Dispersion Aware Digital Backpropagation, Prola et al. (1997): PMD Emulators and Signal Distortion in 2.48-Gb/s IM-DD Lightwave Systems
        DGD_section_stdev = DGD_section_mean / 5
        self.DGD = np.random.default_rng().normal(
            loc = DGD_section_mean,
            scale = DGD_section_stdev,
            size = (self.Nsec, self.Nreal)
        )

    def init_PSP(self):
        """
        Initialise a random rotation of the principal state of polarisation per realisation and fibre section, in Stokes coordinates.
        """
        initialisation_vector = np.random.default_rng().normal(
            size = (self.Nsec, self.Nreal, 4)
        ) # Initialise [cos(theta), a * sin(theta)], Czegledi et al. (2016): Polarization Drift Channel Model for Coherent Fibre-Optic Systems
        initialisation_vector /= np.linalg.norm(initialisation_vector, axis = -1)[..., None] # Normalise

        rotation_angle = np.arccos(initialisation_vector[..., 0, None])
        rotation_axis  = initialisation_vector[..., 1:] / np.sin(rotation_angle)
        self.PSP_stokes = rotation_angle * rotation_axis # ???, notation error in Czegledi et al. ???

    def init_material_strain(self):
        """
        Initialise the fibre with 0 material strain.
        """
        self.material_strain = np.zeros(
            size = (self.Nsec),
            dtype = float
        )

    @override
    def propagate(self, signal: Signal) -> Signal:
        """
        Propagate a polarisation-multiplexed phase-multiplexed signal through the waveplate model.
        The model applies a differential group delay and rotates the principal axes of polarisation in every fibre section.
        Multiple fibre realisations are applied at once.

        Inputs:
        - signal [Signal]: the signal to propagate through the channel in the time domain, shape [Nreal,B,S,P] or [B,S,P] with number of realisations Nreal, batch size B, sample count S, and principal polarisations P=2.

        Outputs:
        - [Signal]: the output signal, shape [Nreal,B,S,P]
        """
        if len(signal.samples.shape) == 3: signal.samples = signal.samples[None] # Add a dimension for the number of fibre realisations
        assert len(signal.samples.shape) == 4, f"Signal must have shape (B,S,P) or (Nreal,B,S,P), but had shape {signal.samples.shape}"
        assert signal.samples.shape[0] in (1, self.Nreal), f"Signal must have first dimension length Nreal ({self.Nreal}), but had length {signal.samples.shape[0]}"

        samples_w = signal.samples_w # Get signal samples in the frequency domain
        w = signal.w # Get sample frequencies in Rad/s

        for section_DGD, section_PSP, section_optical_strain in zip(self.DGD, self.PSP, self.optical_strain):
            # Apply section DGD
            DGD = np.exp(-0.5j * section_DGD[:, *3*(None,)] * (1 + section_optical_strain) * w[*2*(None), :, None] * 1e-12)
            samples_w[..., 0] *= section_DGD
            samples_w[..., 1] *= np.conj(section_DGD)

            # Rotate PSP
            samples_w = np.einsum('rpq,rbsq->rbsp', section_PSP, samples_w)

        return Signal(
            samples = samples_w,
            sample_rate = signal.sample_rate,
            domain = Domain.FREQUENCY
        )

    def to_dict(self) -> dict:
        """
        Represent this waveplate model (and its exact realisations) as a dictionary.

        Outputs:
        - [dict] The dictionary representation of this waveplate model
        """
        return {
            'Lc':    self.Lc,
            'Nsec':  self.Nsec,
            'tau':   self.tau,
            'Nreal': self.Nreal,
            'xi':    self.xi,
            'DGD':   self.DGD,
            'PSP_stokes': self.PSP_stokes,
            'material_strain': self.material_strain,
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
        parameters.set('FIBRE', 'Lc', waveplate_dict['Lc'])
        parameters.set('FIBRE', 'Nsec', waveplate_dict['Nsec'])
        parameters.set('FIBRE', 'tau', waveplate_dict['tau'])
        parameters.set('FIBRE', 'Nreal', waveplate_dict['Nreal'])
        parameters.set('FIBRE', 'xi', waveplate_dict['xi'])

        waveplate = cls(parameters)
        waveplate.DGD = waveplate_dict['DGD']
        waveplate.PSP_stokes = waveplate_dict['PSP_stokes']
        waveplate.material_strain = waveplate_dict['material_strain']

        return waveplate

    @property
    def Lc(self) -> float:
        """
        [float] correlation length of this waveplate model in km.
        """
        return float(self._Lc)

    @Lc.setter
    def Lc(self, value):
        raise AttributeError("The correlation length Lc cannot be changed after creation of the waveplate model.")

    @property
    def Nsec(self) -> int:
        """
        [int] number of fibre sections of this waveplate model.
        """
        return int(self._Nsec)

    @Nsec.setter
    def Nsec(self, value):
        raise AttributeError("The section count Nsec cannot be changed after creation of the waveplate model.")

    @property
    def tau(self) -> float:
        """
        [float] polarisation mode dispersion parameter of this waveplate model in ps/(km ^ 0.5)
        """
        return float(self._tau)

    @tau.setter
    def tau(self, value):
        raise AttributeError("The polarisation mode dispersion parameter tau cannot be changed after creation of the waveplate model")

    @property
    def Nreal(self) -> float:
        """
        [int] realisation count of this waveplate model
        """
        return int(self._Nreal)

    @Nreal.setter
    def Nreal(self, value):
        raise AttributeError("The realisation count Nreal cannot be changed after creation of the waveplate model")

    @property
    def L(self) -> float:
        """
        [float] total length of the fibre in km
        """
        return float(self.Lc * self.Nsec)

    @L.setter
    def L(self, value):
        raise AttributeError("The fibre length L cannot be set directly")

    @property
    def xi(self) -> float:
        """
        [float] fibre photoelasticity constant
        """
        return self._xi

    @xi.setter
    def xi(self, value):
        raise AttributeError("The photoelasticity constant xi cannot be changed after creation of the waveplate model")

    @property
    def DGD(self) -> np.ndarray:
        """
        [float] the differential group delay per section and realisation in ps, shape [Nsec, Nreal]
        """
        return self._DGD

    @DGD.setter
    def DGD(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), f"New DGD value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New DGD array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.Nsec, self.Nreal), f"New DGD array must have shape (self.Nsec ({self.Nsec}), self.Nreal ({self.Nreal})), but had shape {value.shape}"
        self._DGD = value.copy().astype(float)

    @property
    def PSP_stokes(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] principal state of polarisation rotations, expressed in Stokes parameters.
        """
        return self._PSP_stokes

    @PSP_stokes.setter
    def PSP_stokes(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), f"New PSP_stokes value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New PSP_stokes array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.Nsec, self.Nreal), f"New PSP_stokes array must have shape (self.Nsec ({self.Nsec}), self.Nreal ({self.Nreal}), 3), but had shape {value.shape}"
        self._PSP_stokes = value.copy().astype(float)

    @property
    def PSP(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] principal state of polarisation rotations, expressed using rotation matrices
        """
        return sp.linalg.expm(-1j * np.tensordot(self.PSP_stokes, pauli_vector, 1)) # [Nsec, Nreal, 3] * [3, 2, 2] -> [Nsec, Nreal, 2, 2]

    @PSP.setter
    def PSP(self, value):
        raise AttributeError("The principal states of polarisation cannot be set as rotation matrices; set PSP_stokes instead")

    @property
    def material_strain(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] material strain parameter per fibre section.
        """
        return self._material_strain

    @material_strain.setter
    def material_strain(self, value):
        assert isinstance(value, np.ndarray), f"New material_strain value must be type np.ndarray, but was a {type(value)}"
        assert value.dtype in (float, int), f"New material_strain array must contain values of type float, but contained {value.dtype}"
        assert value.shape == (self.Nsec), f"New material_strain array must have shape (self.Nsec ({self.Nsec}),), but had shape {value.shape}"
        self._material_strain = value.copy().astype(float)

    @property
    def optical_strain(self) -> np.ndarray:
        """
        [np.ndarray], dtype [float] optical strain parameter per fibre section.
        """
        return self.xi * self.material_strain

    @optical_strain.setter
    def optical_strain(self, value):
        raise AttributeError("The optical strain cannot be set directly; set material_strain instead")
