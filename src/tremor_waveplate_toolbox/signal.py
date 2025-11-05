"""
A structure representing a signal.
"""
import sys

import numpy as np
try:
    import cupy as cp
except:
    pass

from .constants import Domain, Device

class Signal:
    """
    A class that represents one or multiple signals.
    """
    def __init__(self,
            samples: np.ndarray,
            sample_rate: float,
            domain: Domain = Domain.TIME,
            carrier_wavelength: float = np.inf
        ):
        """
        Create a new Signal.

        Inputs:
        - samples [np.ndarray] or [cp.ndarray]: signal samples, shape [...,S,C] with arbitrary dimensions ..., sample count S, and component count C (e.g. 2 polarisations, 3 seismograms, etc).
        - sample_rate [float]: the sample frequency in Hz.
        - domain [Domain]: domain (time or frequency) in which samples is given.
        - carrier_wavelength [float]: carrier wavelength in nm; inf if the signal is not modulated.

        """
        assert isinstance(domain, Domain), f"domain must be a Domain, but was a {type(domain)}"
        # assert isinstance(device, Device), f"device must be a Device, but was a {type(device)}"
        self._domain = domain
        self._device = Device.CPU if isinstance(samples, np.ndarray) else Device.CUDA
        self.samples = samples
        self.sample_rate = sample_rate
        self.carrier_wavelength = carrier_wavelength

    def copy(self):
        """
        [Signal] return a copy of this signal
        """
        return Signal(
            self.samples.copy(),
            self.sample_rate,
            self.domain,
            self.carrier_wavelength
        )

    def to_domain(self, domain: Domain):
        """
        Transform signal samples to the time- or frequency domain
        """
        if domain == self.domain: return

        match domain:
            case Domain.TIME:
                self.samples = self.xp.fft.ifft(self.samples.copy(), axis = -2, norm = 'ortho')
            case Domain.FREQUENCY:
                self.samples = self.xp.fft.fft(self.samples.copy(), axis = -2, norm = 'ortho')

        self._domain = domain

    def resample(self, new_sample_rate: float):
        """
        Change the sample rate whilst keeping signal duration constant.
        Resamples the signal using zeropadding or truncation in the frequency domain.

        Inputs:
        - new_sample_rate [float]: The new sample rate.
        """
        old_sample_rate  = self.sample_rate
        self.sample_rate = new_sample_rate

        old_signal_length = self.shape[-2]
        new_signal_length = round(self.shape[-2] * self.sample_rate / old_sample_rate)
        
        new_samples_frequency = self.xp.zeros(shape = (*self.shape[:-2], new_signal_length, self.shape[-1]), dtype = complex)
        sample_limit = int(min(old_signal_length, new_signal_length) / 2)
        new_samples_frequency[..., :sample_limit, :]  = self.samples_frequency[..., :sample_limit, :]
        new_samples_frequency[..., -sample_limit:, :] = self.samples_frequency[..., -sample_limit:, :]

        self.samples_frequency = new_samples_frequency

    def __eq__(self, other) -> bool:
        other_device = other.device
        other.to_domain(self.domain)
        other.to_device(self.device)
        if self.samples.shape == other.samples.shape and \
            self.xp.allclose(self.samples, other.samples) and \
            self.xp.isclose(self.sample_rate, other.sample_rate):
            other.to_device(other_device)
            return True
        
        other.to_device(other_device)
        return False

    def to_device(self, device: Device) -> None:
        """
        Move this signal into CPU or GPU memory.

        Inputs:
        - device [Device]: 'CPU' to put the signal onto the CPU and do calculations with numpy. 'CUDA' to put the signal onto a CUDA-enabled device and do calculations with cupy.
        """
        match device:
            case Device.CPU:
                if self.device == Device.CPU: return
                self._device = device
                self.samples = np.array(self.samples.get())

            case Device.CUDA:
                assert 'cupy' in sys.modules, f"Cannot move signal onto GPU without CUDA-enabled installation (see installation instructions)"
                self._device = device
                self.samples = cp.array(self.samples)
            
            case _:
                raise ValueError(f"device must be Device.CPU or Device.CUDA, but was {device}")
    
    @property
    def device(self) -> Device:
        """
        Obtain the device on which the signal currently resides

        Outputs:
        - [Device] CPU or CUDA
        """
        return self._device

    @device.setter
    def device(self, value):
        raise AttributeError("Cannot set device directly; use to_device instead.")

    @property
    def xp(self):
        """
        Return a reference to the numpy (np) or cupy (cp) module, depending on where the Signal currently resides.
        """
        return np if self.device == Device.CPU else cp

    @xp.setter
    def xp(self, value):
        raise AttributeError("Cannot set xp directly; use to_device instead.")

    @property
    def samples(self) -> np.ndarray:
        """
        [np.ndarray, cp.ndarray] The signal samples in the current domain (time or frequency), shape [...,S,C] with arbitrary dimensions ..., signal length S, and component count C (e.g. 2 polarisations, 3 seismograms, etc).
        """
        return self._samples

    @samples.setter
    def samples(self, value):
        assert isinstance(value, (np.ndarray)) or ('cupy' in sys.modules and isinstance(value, cp.ndarray)), f"New samples must have type np.ndarray or cp.ndarray (if cupy is available), but had {type(value)}"
        assert len(value.shape) >= 2, f"New samples must have at least two dimensions ..., S and C, but had only {len(value.shape)}"
        assert value.dtype in (complex, float, int), f"New samples must have datatype complex, but were {value.dtype}"
        # assert '_samples' not in self.__dir__() or self._samples.shape[:-2] + self._samples.shape[-1:] == value.shape[:-2] + value.shape[-1:], f"All new samples dimensions must match the previous samples except dimension -2, but their dimensions were {self._samples.shape} and {value.shape}"
        self._samples = self.xp.array(value.copy()).astype(complex)

    @property
    def domain(self) -> Domain:
        """
        [Domain] The domain of self.samples (TIME or FREQUENCY)
        """
        return self._domain

    @domain.setter
    def domain(self, value: Domain):
        raise AttributeError("Cannot set domain directly; use to_domain instead.")

    @property
    def carrier_wavelength(self):
        """
        [float] the signal carrier wavelength in nm, infinity (np.inf) if the signal is unmodulated.
        """
        return self._carrier_wavelength

    @carrier_wavelength.setter
    def carrier_wavelength(self, value):
        assert isinstance(value, (int, float)), f"New carrier_wavelength must be a float, but was {type(value)}."
        assert value > 0, f"New carrier_wavelength must be larger than 0, but was {value}."
        self._carrier_wavelength = value

    @property
    def carrier_frequency(self):
        """
        [float] the signal carrier frequency in Hz, 0 if the signal is unmodulated.
        """
        return 0 if self.carrier_wavelength == np.inf else sp.constants.speed_of_light / self.carrier_wavelength * 1e9

    @carrier_frequency.setter
    def carrier_frequency(self, value):
        assert isinstance(value, (int, float)), f"New carrier_frequency must be a float, but was {type(value)}."
        assert value >= 0, f"New carrier_frequency must be at least 0, but was {value}."
        assert value < np.inf, f"New carrier_frequency may not be infinity"
        self._carrier_wavelength = sp.constants.speed_of_light / value * 1e9

    @property
    def carrier_frequency_angular(self):
        """
        [float] the signal carrier frequency in Rad/s, 0 if the signal is unmodulated.
        """
        return 2 * np.pi * self.carrier_frequency

    @carrier_frequency_angular.setter
    def carrier_frequency_angular(self, value):
        self.carrier_frequency = value / (2 * np.pi)

    @property
    def samples_time(self) -> np.ndarray:
        """
        [np.ndarray, cp.ndarray] The signal samples in the time domain, shape [...,S,P]
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
        [np.ndarray, cp.ndarray] The signal samples in the frequency domain, shape [...,S,P]
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
        assert value > 0, f"New sample rate must be larger than 0, but was {value}"
        self._sample_rate = float(value)

    @property
    def sample_time(self) -> float:
        """
        [float] sample time in s, inverse of sample rate
        """
        return 1 / self.sample_rate

    @sample_time.setter
    def sample_time(self, value):
        self.sample_rate = 1 / value

    @property
    def time(self) -> np.ndarray:
        """
        [np.ndarray, cp.ndarray] sample times in seconds.
        """
        return self.xp.arange(self.shape[-2]) / self.sample_rate

    @time.setter
    def time(self, value):
        raise AttributeError(f"time cannot be set directly; set sample_time or sample_rate instead")

    @property
    def duration(self):
        """
        [float] entire signal duration in s
        """
        return (self.shape[-2] - 1) * self.sample_time

    @duration.setter
    def duration(self, value):
        raise AttributeError(f"duration cannot be set directly; use interpolated() instead")

    @property
    def frequency(self) -> np.ndarray:
        """
        [np.ndarray, cp.ndarray] Vector with sample frequencies in Hz, corresponding to samples_frequency
        """
        return self.xp.fft.fftfreq(
            n = self.shape[-2],
            d = self.sample_time
        )

    @frequency.setter
    def frequency(self, value):
        raise AttributeError(f"frequency cannot be set directly; set sample_time or sample_rate instead")

    @property
    def frequency_angular(self) -> np.ndarray:
        """
        [np.ndarray, cp.ndarray] Vector with sample frequencies in Rad/s, corresponding to samples_frequency
        """
        return 2 * self.xp.pi * self.frequency

    @frequency_angular.setter
    def frequency_angular(self, value):
        raise AttributeError(f"frequency_angular cannot be set directly; set sample_time or sample_rate instead")

    @property
    def frequency_angular_digital(self) -> np.ndarray:
        """
        [np.ndarray, cp.ndarray] Vector with sample frequencies in Rad/sample, corresponding to samples_frequency
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
        [np.ndarray, cp.ndarray] Signal energy, shape [...]
        """
        return self.xp.sum(self.xp.linalg.norm(self.samples, axis = -1) ** 2, axis = -1)

    @energy.setter
    def energy(self, value):
        raise AttributeError(f"energy cannot be set directly; set samples_time or samples_frequency instead")

    @property
    def power_dBm(self) -> np.ndarray:
        """
        [np.ndarray, cp.ndarray] Signal power in dBm, shape [...]
        """
        return 10 * self.xp.log10(1000 * self.power_W)

    @power_dBm.setter
    def power_dBm(self, value):
        self.power_W = 0.001 * 10 ** (value / 10)

    @property
    def power_W(self) -> float:
        """
        [np.ndarray, cp.ndarray] Signal power in W, shape [...]
        """
        return self.energy / self.shape[-2]

    @power_W.setter
    def power_W(self, value):
        self.energy = value * self.shape[-2]
