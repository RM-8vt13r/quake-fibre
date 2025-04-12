"""
A structure representing a signal.
"""

import numpy as np

from .constants import Domain

class Signal:
    """
    A class that represents one or multiple signals.
    """
    def __init__(self,
            samples: np.ndarray,
            sample_rate: float,
            domain: Domain = Domain.TIME
        ):
        """
        Create a new Signal.

        Inputs:
        - samples [np.ndarray]: signal samples, shape [R,B,S,P] with fibre realisations R, batch size B, sample count S, and principal polarisation components P = 2.
        - sample_rate [float]: the sample frequency in Hz.
        - domain [Domain]: domain (time or frequency) in which samples is given.
        """
        self._domain = domain
        self.samples = samples
        self.sample_rate = sample_rate

    def copy(self):
        """
        [Signal] return a copy of this signal
        """
        return Signal(
            self.samples.copy(),
            self.sample_rate,
            self.domain
        )

    def to_domain(self, domain: Domain):
        """
        Transform signal samples to the time- or frequency domain
        """
        if domain == self.domain: return

        match domain:
            case Domain.TIME:
                self.samples = np.fft.ifft(self.samples.copy(), axis = -2, norm = 'ortho')
            case Domain.FREQUENCY:
                self.samples = np.fft.fft(self.samples.copy(), axis = -2, norm = 'ortho')

        self._domain = domain

    @property
    def samples(self) -> np.ndarray:
        """
        [np.ndarray] The signal samples in the current domain (time or frequency), shape [R,B,S,P] with fibre realisations R, batch size B, signal length S, and principal polarisation components P = 2.
        """
        return self._samples

    @samples.setter
    def samples(self, value):
        assert isinstance(value, np.ndarray), f"New samples must have type np.ndarray, but had {type(value)}"
        assert len(value.shape) == 4, f"New samples must have four dimensions R, B, S and P, but had only {len(value.shape)}"
        assert value.shape[-1] == 2, f"New samples must have two polarisation components on the final dimension, but had {value.shape[-1]}"
        assert value.dtype in (complex, float, int), f"New samples must have datatype complex, but were {value.dtype}"
        self._samples = value.copy().astype(complex)

    @property
    def domain(self) -> Domain:
        """
        [Domain] The domain of self.samples (TIME or FREQUENCY)
        """
        return self._domain

    @domain.setter
    def domain(self, value: Domain):
        raise AttributeError("Cannot set domain directly; use to_domain instead.")
        # assert isinstance(value, Domain), f"New domain must have type Domain, but had type {type(value)}"
        # self._domain = value

    @property
    def samples_time(self) -> np.ndarray:
        """
        [np.ndarray] The signal samples in the time domain, shape [R,B,S,P]
        """
        self.to_domain(Domain.TIME)
        return self.samples

    @samples_time.setter
    def samples_time(self, value):
        self.samples = value
        self._domain = Domain.TIME

    @property
    def samples_frequency(self) -> np.ndarray:
        """
        [np.ndarray] The signal samples in the frequency domain, shape [R,B,S,P]
        """
        self.to_domain(Domain.FREQUENCY)
        return self.samples

    @samples_frequency.setter
    def samples_frequency(self, value):
        self.samples = value
        self._domain = Domain.FREQUENCY

    @property
    def sample_rate(self) -> float:
        """
        [float] Signal sample rate in Hz
        """
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        assert isinstance(value, (float, int)), f"New sample rate must have type float, but was {type(value)}"
        assert value > 0, f"New sample rate must be larger than 0, ut was {type(value)}"
        self._sample_rate = float(value)

    @property
    def sample_time(self) -> float:
        """
        [float] sample time in s, inverse of sample rate
        """
        return 1 / self.sample_rate

    @sample_time.setter
    def sample_time(self, value):
        self.sample_rate = 1  / value

    @property
    def time(self) -> np.ndarray:
        """
        [np.ndarray] sample times in seconds.
        """
        return np.arange(self.shape[-2]) / self.sample_rate

    @time.setter
    def time(self, value):
        raise AttributeError(f"time cannot be set directly; set sample_time or sample_rate instead")

    @property
    def frequency(self) -> np.ndarray:
        """
        [np.ndarray] Vector with sample frequencies in Hz, corresponding to samples_frequency
        """
        return np.fft.fftfreq(
            n = self.shape[-2],
            d = self.sample_time
        )

    @frequency.setter
    def frequency(self, value):
        raise AttributeError(f"frequency cannot be set directly; set sample_time or sample_rate instead")

    @property
    def frequency_angular(self) -> np.ndarray:
        """
        [np.ndarray] Vector with sample frequencies in Rad/s, corresponding to samples_frequency
        """
        return 2 * np.pi * self.frequency

    @frequency_angular.setter
    def frequency_angular(self, value):
        raise AttributeError(f"frequency_angular cannot be set directly; set sample_time or sample_rate instead")

    @property
    def frequency_angular_digital(self) -> np.ndarray:
        """
        [np.ndarray] Vector with sample frequencies in Rad/sample, corresponding to samples_frequency
        """
        return self.sample_time * self.frequency_angular

    @frequency_angular_digital.setter
    def frequency_angular_digital(self, value):
        raise AttributeError(f"frequency_angular_digital cannot be set directly; set sample_time or sample_rate instead")

    @property
    def shape(self) -> tuple:
        """
        [tuple] Shape of the signal samples
        """
        return self.samples.shape

    @shape.setter
    def shape(self, value):
        raise AttributeError(f"shape cannot be set directly; set samples_time or samples_frequency instead")

    @property
    def energy(self) -> np.ndarray:
        """
        [np.ndarray] Signal energy, shape [R,B]
        """
        return np.sum(np.linalg.norm(self.samples, axis = -1) ** 2, axis = -1)

    @energy.setter
    def energy(self, value):
        raise AttributeError(f"energy cannot be set directly; set samples_time or samples_frequency instead")

    @property
    def power_dBm(self) -> np.ndarray:
        """
        [np.ndarray] Signal power in dBm, shape [R,B]
        """
        return 10 * np.log10(1000 * self.power_W)

    @power_dBm.setter
    def power_dBm(self, value):
        self.power_W = 0.001 * 10 ** (value / 10)

    @property
    def power_W(self) -> float:
        """
        [np.ndarray] Signal power in W, shape [R,B]
        """
        return self.energy / self.shape[-2]

    @power_W.setter
    def power_W(self, value):
        self.energy = value * self.shape[-2]
