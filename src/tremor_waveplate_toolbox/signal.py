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
        - samples [np.ndarray]: signal samples, shape [...,S,P] with arbitrary dimensions ..., sample count S, and principal polarisation components P = 2.
        - sample_rate [float]: the sample frequency in Hz.
        - domain [Domain]: domain (time or frequency) in which samples is given.
        """
        match domain:
            case Domain.TIME: self.samples_t = samples
            case Domain.FREQUENCY: self.samples_f = samples

        self.sample_rate = sample_rate

    def copy(self):
        """
        [Signal] return a copy of this signal
        """
        return Signal(
            self.samples_t.copy(),
            self.sample_rate
        )

    @property
    def samples_t(self) -> np.ndarray:
        """
        [np.ndarray] The signal samples, shape [... ,S,P] with arbitrary dimensions ..., sample count S, and principal polarisation components P=2.
        """
        return self._samples_t

    @samples_t.setter
    def samples_t(self, value):
        assert isinstance(value, np.ndarray), f"New samples must have type np.ndarray, but had {type(value)}"
        assert len(value.shape) >= 2, f"New samples must have at least two dimensions S and P, but had only {len(value.shape)}"
        assert value.shape[-1] == 2, f"New samples must have two polarisation components on the final dimension, but had {value.shape[-1]}"
        assert value.dtype in (complex, float, int), f"New samples must have datatype complex, but were {value.dtype}"
        self._samples_t = value.copy().astype(complex)

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
    def t(self) -> np.ndarray:
        """
        [np.ndarray] sample times in seconds.
        """
        return np.arange(self.shape[-2]) / self.sample_rate

    @t.setter
    def t(self, value):
        raise AttributeError(f"t cannot be set directly; set sample_time or sample_rate instead")

    @property
    def samples_f(self) -> np.ndarray:
        """
        [np.ndarray] samples in the frequency domain
        """
        return np.fft.fft(self.samples_t, axis = -2)

    @samples_f.setter
    def samples_f(self, value):
        self.samples_t = np.fft.ifft(value, axis = -2)

    @property
    def samples_w(self) -> np.ndarray:
        """
        [np.ndarray] samples in the frequency domain
        """
        return self.samples_f

    @samples_w.setter
    def samples_w(self, value):
        self.samples_f = value

    @property
    def f(self) -> np.ndarray:
        """
        [np.ndarray] Vector with sample frequencies in Hz, corresponding to samples_f
        """
        return np.fft.fftfreq(
            n = self.shape[-2],
            d = self.sample_time
        )

    @f.setter
    def f(self, value):
        raise AttributeError(f"f cannot be set directly; set sample_time or sample_rate instead")

    @property
    def w(self) -> np.ndarray:
        """
        [np.ndarray] Vector with sample frequencies in Rad/s, corresponding to samples_f
        """
        return 2 * np.pi * self.sample_time * self.f

    @w.setter
    def w(self, value):
        raise AttributeError(f"w cannot be set directly; set sample_time or sample_rate instead")

    @property
    def shape(self) -> tuple:
        """
        [tuple] Shape of the signal samples
        """
        return self.samples_t.shape

    @shape.setter
    def shape(self, value):
        raise AttributeError(f"shape cannot be set directly; set samples_t, samples_f or samples_w instead")

    @property
    def energy(self) -> np.ndarray:
        """
        [np.ndarray] Signal energy, shape [...] (see samples_t)
        """
        return np.sum(np.linalg.norm(self.samples_t, axis = -1) ** 2, axis = -1)

    @energy.setter
    def energy(self, value):
        raise AttributeError(f"energy cannot be set directly; set samples_t, samples_f or samples_w instead")

    @property
    def power_dBm(self) -> np.ndarray:
        """
        [np.ndarray] Signal power in dBm, shape [...] (see samples_t)
        """
        return 10 * np.log10(1000 * self.power_W)

    @power_dBm.setter
    def power_dBm(self, value):
        self.power_W = 0.001 * 10 ** (value / 10)

    @property
    def power_W(self) -> float:
        """
        [np.ndarray] Signal power in W, shape [...] (see samples_t)
        """
        return self.energy / self.shape[-2]

    @power_W.setter
    def power_W(self, value):
        self.energy = value * self.shape[-2]
