"""
Pulses for pulseshaping
"""

from abc import ABC, abstractmethod
from typing import override

import numpy as np

from .constants import Domain
from .signal import Signal

class Pulse(ABC):
    """
    An abstract class representing a pulseshape.
    Create specific pulseshapes by subclassing this class and overriding the pulse_time and/or pulse_frequency methods.
    """
    def __init__(self, symbol_rate: float, domain: Domain):
        """
        Initialise the pulseshape.

        Inputs:
        - symbol_rate [float]: the symbol rate in Hz
        - domain [Domain]: the domain in which the pulseshape should be applied
        """
        self.symbol_rate = symbol_rate
        self._domain = domain

    def __call__(self, signal: Signal) -> Signal:
        """
        See modulate()
        """
        return self.modulate(signal)

    def modulate(self, signal: Signal) -> Signal:
        """
        Convolve a signal with this pulse.
        Pulseshaping is done in the frequency- or time domain, based on the pulse's 'domain' variable.

        Inputs:
        - signal [Signal]: the signal to modulate.

        Outputs:
        - signal [Signal]: the pulse amplitude-modulated signal.
        """
        match self.domain:
            case Domain.TIME: return self.modulate_time(signal)
            case Domain.FREQUENCY: return self.modulate_frequency(signal)
            case _: raise ValueError(f"domain must be time or frequency, but was {self.domain}")

    def modulate_time(self, signal: Signal) -> Signal:
        """
        Convolve a signal with this pulse in the time domain.

        Inputs:
        - signal [Signal]: the signal to modulate, shape [R,B,S,P] with number of fibre realisations R or R = 1, batch size B, number of samples S, and number of polarisations P = 2.

        Outputs:
        - signal [Signal]: the pulse amplitude-modulated signal.
        """
        raise NotImplementedError()

    def modulate_frequency(self, signal: Signal) -> Signal:
        """
        Convolve a signal with this pulse in the frequency domain.

        Inputs:
        - signal [Signal]: the signal to modulate, shape [R,B,S,P] with number of fibre realisations R or R = 1, batch size B, number of samples S, and number of polarisations P = 2.

        Outputs:
        - signal [Signal]: the pulse amplitude-modulated signal.
        """
        return Signal(
            samples = signal.samples_frequency * self.pulse_frequency(signal.frequency)[None, None, :, None] * np.sqrt(signal.sample_rate),
            sample_rate = signal.sample_rate,
            domain = Domain.FREQUENCY,
            carrier_wavelength = signal.carrier_wavelength
        )

    def modulate_time(self, signal: Signal) -> Signal:
        raise NotImplementedError()

    @abstractmethod
    def pulse_time(self, time: np.ndarray) -> np.ndarray:
        """
        Generate the pulse in the time domain at the time values in array time.

        Inputs:
        - time [np.ndarray], dtype [float]: time vector in s, at which values to calculate the pulse.

        Outputs:
        - [np.ndarray] pulse in the time domain at the times from time.
        """
        raise NotImplementedError

    @abstractmethod
    def pulse_frequency(self, frequency: np.ndarray) -> np.ndarray:
        """
        Generate the pulse in the frequency domain at the frequency values in array frequency.

        Inputs:
        - frequency [np.ndarray], dtype [float]: frequency vector in Hz, at which values to calculate the pulse in the frequency domain.

        Outputs:
        - [np.ndarray] pulse in the frequency domain at the frequencies from frequency.
        """
        raise NotImplementedError

    # def pulse_w(self, w: np.ndarray, upsample_factor: int = 1) -> np.ndarray:
    #     """
    #     Generate the pulse in the frequency domain at the frequency values in array w.
    #
    #     Inputs:
    #     - w [np.ndarray], dtype [float]: frequency vector in rad/s, at which values to calculate the pulse in the frequency domain.
    #     - upsample_factor [int]: upsampling factor to determine the sample rate
    #
    #     Outputs:
    #     - [np.ndarray] pulse in the frequency domain at the frequencies from w.
    #     """
    #     return self.pulse_f(w / (2 * np.pi * self.symbol_time))

    @property
    def domain(self) -> Domain:
        """
        [Domain] the domain (time or frequency) in which modulation with this pulse takes place
        """
        return self._domain

    @domain.setter
    def domain(self, value):
        assert isinstance(value, Domain), f"domain must have type Domain, but was {type(value)}"
        self._domain = value

    @property
    def symbol_rate(self) -> float:
        """
        [float] the symbol rate in Hz.
        """
        return self._symbol_rate

    @symbol_rate.setter
    def symbol_rate(self, value):
        assert isinstance(value, (float, int)), f"symbol_rate must have type float, but was {type(value)}"
        assert value > 0, f"symbol_rate must be positive, but was {value}"
        self._symbol_rate = float(value)

    @property
    def symbol_time(self) -> float:
        """
        [float] duration of a symbol in s
        """
        return 1 / self.symbol_rate

    @symbol_time.setter
    def symbol_time(self, value):
        self.symbol_rate = 1 / value

class Sinc(Pulse):
    """
    A sinc pulse
    """
    def __init__(self, symbol_rate: float):
        """
        Initialise the sinc pulse.

        Inputs:
        - symbol_rate [float]: the symbol rate in Hz
        - rolloff [float]: the rolloff factor
        """
        super().__init__(symbol_rate, Domain.FREQUENCY)

    @override
    def pulse_time(self, time: np.ndarray) -> np.ndarray:
        return np.sinc(time * self.symbol_rate) * np.sqrt(self.symbol_rate)

    @override
    def pulse_frequency(self, frequency: np.ndarray) -> np.ndarray:
        return np.where(np.abs(frequency) <= self.symbol_rate / 2, 1 / np.sqrt(self.symbol_rate), 0.)

class RootRaisedCosine(Pulse):
    """
    A root-raised cosine pulse
    """
    def __init__(self, symbol_rate: float, rolloff: float):
        """
        Initialise the root-raised cosine pulse.

        Inputs:
        - symbol_rate [float]: the symbol rate in Hz
        - rolloff [float]: the rolloff factor
        """
        super().__init__(symbol_rate, Domain.FREQUENCY)
        self.rolloff = rolloff

    @override
    def pulse_time(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @override
    def pulse_frequency(self, frequency: np.ndarray) -> np.ndarray:
        plateau_mask = np.abs(frequency) <= (1 - self.rolloff) * self.symbol_rate / 2
        side_mask   = (np.abs(frequency) <= (1 + self.rolloff) * self.symbol_rate / 2) & (~plateau_mask)

        rcos = np.zeros_like(frequency, dtype = float)
        rcos[plateau_mask] = 1
        if np.sum(side_mask): rcos[side_mask] = (1 + np.cos(np.pi / (self.rolloff * self.symbol_rate) * (np.abs(frequency[side_mask]) - (1 - self.rolloff) * self.symbol_rate / 2))) / 2

        rrcos = np.sqrt(rcos / self.symbol_rate)

        return rrcos

    @property
    def rolloff(self) -> float:
        """
        The root-raised cosine rolloff factor
        """
        return self._rolloff

    @rolloff.setter
    def rolloff(self, value: float):
        assert isinstance(value, (float, int)), f"rolloff must have type float, but was {type(value)}"
        assert value >= 0, f"rolloff must be nonnegative, but was {value}"
        assert value <= 1, f"rolloff must be <= 1, but was {value}"
        self._rolloff = float(value)

SINC = Sinc
RRCOS = RootRaisedCosine
