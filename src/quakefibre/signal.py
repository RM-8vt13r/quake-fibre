"""
A structure representing a signal.
"""
import sys
import os
import logging

import numpy as np
import scipy as sp
try:
    import cupy as cp
except:
    pass
import netCDF4 as nc

from .constants import Domain, Device, Dimension, Gain
from .utilities import dB2linear, linear2dB
from .dataset import create_attributes, create_dimensions, create_variables, read_variable, write_variable

logger = logging.getLogger()

class Signal:
    """
    A class that represents one or multiple signals.
    """
    def __init__(self,
            samples: np.ndarray,
            sample_rate: float,
            sample_axis: int = -2,
            domain: Domain = Domain.TIME,
            carrier_wavelength: float = np.inf
        ):
        """
        Create a new Signal.

        Inputs:
        - samples [np.ndarray] or [cp.ndarray]: signal samples, shape [...,S,*C] with arbitrary dimensions ..., sample count S, and component counts *C (e.g. 2 polarisations, 3 seismograms, [2, 2] matrix elements, etc).
        - sample_rate: sample rate in samples per second.
        - sample_axis [int]: the index of axis S in the shape of samples
        - domain [Domain]: domain (time or frequency) in which samples is given.
        - carrier_wavelength [float]: carrier wavelength in nm; inf if the signal is not modulated.
        """
        assert isinstance(domain, (Domain, str)), f"domain must be a Domain, but was a {type(domain)}"
        if isinstance(domain, str):
            self._domain = Domain[domain]
        else:
            self._domain = domain
        self._device = Device.CPU if isinstance(samples, np.ndarray) else Device.CUDA
        self.samples = samples
        self.sample_rate = sample_rate
        self.sample_axis = sample_axis
        self.carrier_wavelength = carrier_wavelength

    def copy(self):
        """
        [Signal] return a copy of this signal
        """
        return Signal(
            samples     = self.samples.copy(),
            sample_rate = self.sample_rate,
            sample_axis = self.sample_axis,
            domain      = self.domain,
            carrier_wavelength = self.carrier_wavelength
        )

    def to_domain(self, domain: Domain):
        """
        Transform signal samples to the time- or frequency domain
        """
        if domain == self.domain: return


        if not np.issubdtype(self.samples.dtype, np.complexfloating):
            logger.warning("(Inverse) Fourier transform cast real-valued Signal to complex")

        match domain:
            case Domain.TIME:
                self.samples = self.xp.fft.ifft(self.samples, axis = self.sample_axis, norm = 'ortho')
            case Domain.FREQUENCY:
                self.samples = self.xp.fft.fft(self.samples, axis = self.sample_axis, norm = 'ortho')

        self._domain = domain

    def resampled(self, new_sample_rate: float):
        """
        Return a copy of the signal, changing the sample rate whilst keeping signal duration constant.
        Resamples the signal using zeropadding or truncation in the frequency domain.

        Inputs:
        - new_sample_rate [float]: The new sample rate.

        Outputs:
        - [Signal] the new Signal
        """
        old_signal_length = self.shape[self.sample_axis]
        new_signal_length = round(self.shape[self.sample_axis] * new_sample_rate / self.sample_rate)
        
        new_samples_frequency = self.xp.zeros(shape = (*self.shape[:self.sample_axis], new_signal_length, *self.shape[self.sample_axis_nonnegative + 1:]), dtype = complex)
        sample_limit = int(min(old_signal_length, new_signal_length) / 2)
        new_samples_frequency[..., :sample_limit, *(slice(None),) * -(self.sample_axis_negative + 1)]  = self.samples_frequency[..., :sample_limit, *(slice(None),) * -(self.sample_axis_negative + 1)]
        new_samples_frequency[..., -sample_limit:, *(slice(None),) * -(self.sample_axis_negative + 1)] = self.samples_frequency[..., -sample_limit:, *(slice(None),) * -(self.sample_axis_negative + 1)]

        return Signal(
                samples = new_samples_frequency,
                sample_rate = new_sample_rate,
                sample_axis = self.sample_axis,
                domain = Domain.FREQUENCY,
                carrier_wavelength = self.carrier_wavelength
            )

    def __eq__(self, other) -> bool:
        other_device = other.device
        other.to_domain(self.domain)
        other.to_device(self.device)
        if self.shape == other.shape and \
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
                if self.device == Device.CPU:
                    # logger.warning("Attempted to move signal into CPU memory, but it already resided there.")
                    return
                    
                self._device = device
                self.samples = np.array(self.samples.get())

            case Device.CUDA:
                # assert 'cupy' in sys.modules, f"Cannot move signal onto GPU without CUDA-enabled installation (see installation instructions)"
                if 'cupy' not in sys.modules:
                    logger.warning("Cannot move signal onto GPU without CUDA-enabled installation (see installation instructions) -> keeping signal in CPU memory")
                    return
                    
                self._device = device
                self.samples = cp.array(self.samples)
                    
            case _:
                raise ValueError(f"device must be Device.CPU or Device.CUDA, but was {device}")
    
    def invite_array(self, array: np.ndarray):
        """
        Move an array into the same memory device as this Signal.

        Inputs:
        - array [np.ndarray, cp.ndarray, list, tuple]: the array to move to the same device as this Signal

        Outputs:
        - [np.ndarray, cp.ndarray]: the moved array
        """
        assert isinstance(array, (np.ndarray, list, tuple, int, float, complex, np.number)) or ('cupy' in sys.modules and isinstance(array, (cp.ndarray, cp.number))), f"array must be a list, tuple, np.ndarray, or cp.ndarray, but was a {type(array)}"

        if 'cupy' in sys.modules and self.device == Device.CPU and isinstance(array, cp.ndarray):
            # Copy array from GPU to CPU memory
            return array.get()

        # Three remaining situations:
        # 1) array is not an array -> convert it to one
        # 2) signal is on GPU -> copy array over from CPU or another GPU
        # 3) CUDA is not enabled -> everything is on CPU already, but may need to be converted from list to array
        return self.xp.array(array)

    def save(self, dataset: nc.Dataset, step_starts: dict = {}, allow_attribute_overwrite: bool = False, compression: str = 'zlib', compression_level: int = 4) -> nc.Dataset:
        """
        Save this Signal in a file as a netCDF4 Dataset

        Inputs:
        - dataset [nc.Dataset]: Dataset to save in.
        - step_starts [dict]: {dimension [int, str]: step_start [int]} pairs. For each key, treat dimension index 0 in this Signal as index step_start in the netCDF file. This allows e.g. the gradual saving of a large Signal, by appending multiple smaller Signals.
        - allow_attribute_overwrite [bool]: If False, throws an error when you attempt to overwrite an existing dataset attribute with a new value.
        - compression [str]: Compression algorithm to use for saving. If None, save uncompressed data. See https://unidata.github.io/netcdf4-python/#efficient-compression-of-netcdf-variables
        - compression_level [int]: How aggressively to compress. 0 = no compression. 1 = mild compression (fast, but large file). 9 = aggressive compression (slow, but small file)

        Outputs:
        - [nc.Dataset]: the Dataset
        """
        original_device = self.device
        self.to_device(Device.CPU)

        dataset_dimensions = [Dimension.DIMENSION.name + str(index + 1) for index in range(self.sample_axis_nonnegative)] + [Dimension.SAMPLES.name] + [Dimension.CHANNELS.name + str(component + 1) for component in range(-1 - self.sample_axis_negative)]
        create_dimensions(dataset, dataset_dimensions)
        create_variables(dataset, ('samples_real', 'samples_imaginary'), 'f4', dataset.dimensions, compression, compression_level)
        write_variable(dataset, 'samples_real', self.samples.real, step_starts)
        write_variable(dataset, 'samples_imaginary', self.samples.imag, step_starts)
        create_attributes(dataset, ('sample_rate', 'sample_axis', 'domain', 'carrier_wavelength'), (self.sample_rate, self.sample_axis_nonnegative, self.domain.name, self.carrier_wavelength), allow_attribute_overwrite)
        dataset.sync()
        
        self.to_device(original_device)

        return dataset

    @classmethod
    def load(cls, dataset: nc.Dataset, step_starts: dict = {}, step_stops: dict = {}):
        """
        Load a Signal from a netCDF4 Dataset

        inputs:
        - dataset [nc.Dataset]: the Dataset to load the Signal from
        - step_starts [dict]: {dimension [int, str]: step_start [int]} pairs. For each key, treat dimension index 0 in this Signal as index step_start in the netCDF file. This allows the partial loading of a large Signal.
        - step_stops [dict]: {dimension [int, str]: step_stop [int]} pairs. For each key, treat dimension index -1 + 1 in this Signal as index step_stop in the netCDF file. This allows the partial loading of a large Signal.
        
        outputs:
        - [Signal]: the loaded Signal
        """
        return Signal(
                samples = np.array(read_variable(dataset, 'samples_real', step_starts, step_stops)) + 1j * np.array(read_variable(dataset, 'samples_imaginary', step_starts, step_stops)),
                **{key: dataset.getncattr(key) for key in dataset.ncattrs()}
            )

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
        [np.ndarray, cp.ndarray] The signal samples in the current domain (time or frequency), shape [...,S,*C] with arbitrary dimensions ..., signal length S, and component count C (e.g. 2 polarisations, 3 seismograms, [2, 2] matrix elements etc).
        """
        return self._samples

    @samples.setter
    def samples(self, value):
        """
        Set the samples of the signal in its current domain, shape [..., S, *C] where ... can be anything, S is the number of samples on axis self.sample_axis, and *C is any number of channels.
        Note: value is not copied for the signal! Rather, a view is stored.
        """
        assert isinstance(value, (np.ndarray)) or ('cupy' in sys.modules and isinstance(value, cp.ndarray)), f"New samples must have type np.ndarray or cp.ndarray (if cupy is available), but had {type(value)}"
        assert len(value.shape) >= 1, f"New samples must have at least one dimension ..., S and *C, but had only {len(value.shape)}"
        assert np.issubdtype(value.dtype, np.number), f"New samples must have datatype complex or float, but were {value.dtype}"
        if hasattr(self, '_samples') and not np.issubdtype(value.dtype, self.samples.dtype) and not np.issubdtype(self.samples.dtype, value.dtype):
            logger.warning(f"Setting samples changed {self.samples.dtype} Signal to {value.dtype}")
        # assert '_samples' not in self.__dir__() or self.shape[:-2] + self.shape[-1:] == value.shape[:-2] + value.shape[-1:], f"All new samples dimensions must match the previous samples except dimension -2, but their dimensions were {self.shape} and {value.shape}"
        self._samples = self.xp.array(value)

    @property
    def sample_axis(self):
        """
        Returns the index of axis S
        """
        return self._sample_axis

    @sample_axis.setter
    def sample_axis(self, value):
        assert isinstance(value, (int, np.integer)), f"sample_axis must be an int, but was a {type(value)}"
        assert value >= -len(self.shape), f"sample_axis must be at least -the number of sample dimensions (>= {-len(self.shape)}), but was {value}"
        assert value < len(self.shape), f"sample_axis must be less than the number of sample dimensions (< {len(self.shape)}), but was {value}"
        self._sample_axis = value

    @property
    def sample_axis_nonnegative(self):
        """
        Returns the index of axis S, enforcing that it's >= 0
        """
        return self._sample_axis % len(self.shape)

    @sample_axis_nonnegative.setter
    def sample_axis_nonnegative(self, value):
        assert value >= 0, f"sample_axis_nonnegative must be >= 0, but was {value}"
        self.sample_axis = value

    @property
    def sample_axis_negative(self):
        """
        Returns the index of axis S, enforcing that it's >= 0
        """
        return self.sample_axis_nonnegative - len(self.shape)

    @sample_axis_negative.setter
    def sample_axis_negative(self, value):
        assert value < 0, f"sample_axis_negative must be < 0, but was {value}"
        self.sample_axis = value

    @property
    def component_axes_count(self):
        """
        The number of component axes
        """
        return -self.sample_axis_negative - 1

    @component_axes_count.setter
    def component_axes_count(self, value):
        raise AttributeError("component_axes_count cannot be set directly; make a new Signal instead")

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
        assert isinstance(value, (int, np.integer, float, np.floating)), f"New carrier_wavelength must be a float, but was {type(value)}."
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
        assert isinstance(value, (int, np.integer, float, np.floating)), f"New carrier_frequency must be a float, but was {type(value)}."
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
        [np.ndarray, cp.ndarray] The signal samples in the time domain, shape [...,S,*C]
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
        [np.ndarray, cp.ndarray] The signal samples in the frequency domain, shape [...,S,*C]
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
        assert isinstance(value, (int, np.integer, float, np.floating)), f"New sample rate must have type float, but was {type(value)}"
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
        return self.xp.arange(self.shape[self.sample_axis]) / self.sample_rate

    @time.setter
    def time(self, value):
        raise AttributeError(f"time cannot be set directly; set sample_time or sample_rate instead")

    @property
    def duration(self):
        """
        [float] entire signal duration in s
        """
        return self.shape[self.sample_axis] * self.sample_time

    @duration.setter
    def duration(self, value):
        raise AttributeError(f"duration cannot be set directly; set sample_time, sample_rate, or samples instead")

    @property
    def frequency(self) -> np.ndarray:
        """
        [np.ndarray, cp.ndarray] Vector with sample frequencies in Hz, corresponding to samples_frequency
        """
        return self.xp.fft.fftfreq(
            n = self.shape[self.sample_axis],
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
    def sample_bandwidth(self):
        """
        [float] The bandwidth in Hz of a single Signal frequency sample.
        """
        return self.bandwidth / self.sample_count

    @sample_bandwidth.setter
    def sample_bandwidth(self, value):
        raise AttributeError(f"sample_bandwidth cannot be set directly; set sample_time or sample_rate and samples_time or samples_frequency instead")

    @property
    def sample_bandwidth_angular(self):
        """
        [float] The bandwidth in Rad/s of a single Signal frequency sample.
        """
        return 2 * np.pi * self.sample_bandwidth

    @sample_bandwidth_angular.setter
    def sample_bandwidth_angular(self, value):
        raise AttributeError(f"sample_bandwidth_angular cannot be set directly; set sample_time or sample_rate and samples_time or samples_frequency instead")
        
    @property
    def bandwidth(self):
        """
        [float] The bandwidth in Hz that this Signal occupies.
        """
        return self.sample_rate

    @bandwidth.setter
    def bandwidth(self, value):
        raise AttributeError(f"bandwidth cannot be set directly; set sample_time or sample_rate and samples instead")

    @property
    def bandwidth_angular(self):
        """
        [float] The bandwidth in Rad/s that this Signal occupies.
        """
        return 2 * np.pi * self.bandwidth

    @bandwidth_angular.setter
    def bandwidth_angular(self, value):
        raise AttributeError(f"bandwidth_angular cannot be set directly; set sample_time or sample_rate and samples instead")

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
    def sample_count(self) -> int:
        """
        [int] The number of samples along the time/frequency axis.
        """
        return self.shape[self.sample_axis_negative]

    @sample_count.setter
    def sample_count(self, value):
        raise AttributeError(f"sample_count cannot be set directly; set samples_time or samples_frequency instead")

    @property
    def energy(self) -> np.ndarray:
        """
        [np.ndarray, cp.ndarray] Signal energy, shape [...]
        """
        return self.xp.linalg.norm(self.samples, axis = tuple(range(self.sample_axis_negative, 0))) ** 2

    @energy.setter
    def energy(self, value):
        raise AttributeError(f"energy cannot be set directly; set samples_time or samples_frequency instead")

    @property
    def power_dBm(self) -> np.ndarray:
        """
        [np.ndarray, cp.ndarray] Signal power in dBm, shape [...]
        """
        return linear2dB(1000 * self.power_W, Gain.POWER)

    @power_dBm.setter
    def power_dBm(self, value):
        self.power_W = 0.001 * dB2linear(value, Gain.POWER)

    @property
    def power_W(self) -> float:
        """
        [np.ndarray, cp.ndarray] Signal power in W, shape [...]
        """
        return self.energy / self.sample_count

    @power_W.setter
    def power_W(self, value):
        self.energy = value * self.sample_count
