"""
A class to receive optical signals
"""

from configparser import ConfigParser

import numpy as np

from .signal import Signal
from . import constellation as const
from . import pulse

class Receiver:
    """
    A class that acts as a receiver.
    It downsamples a signal stream with antialiasing.
    """
    def __init__(self, parameters: ConfigParser):
        """
        Create a new Receiver.

        Required entries in parameters['TRANSCEIVER']:
        - power [float]: Transmission power in dBm
        - baud_rate [float]: symbol rate in Hz
        - filter [str]: 'SINC' or 'RRCOS' for sinc or root-raised cosine antialiasing filters
        - filter_parameter [object]: any parameter(s) to define the filter, such as rolloff factor
        - upsample_factor [int]: samples per symbol
        """
        self.filter            = [parameters.get('TRANSCEIVER', 'filter'), (parameters.getfloat('TRANSCEIVER', 'baud_rate'), parameters.getfloat('TRANSCEIVER', 'filter_parameter'))]
        self.power_dBm         = parameters.getfloat('TRANSCEIVER', 'power')
        self.downsample_factor = int(parameters.getfloat('TRANSCEIVER', 'upsample_factor'))

    def __call__(self, samples: Signal):
        """
        See receive_samples()
        """
        return self.receive_samples(samples)

    def receive_samples(self, samples: Signal):
        """
        Attenuate, anti-aliasing filter, and downsample a given sequence of samples

        Inputs:
        - samples [Signal]: the sequence of samples to receive, shape [B,S,2] where B is batch size, S is the sequence length, and the last dimension indexes two orthogonal polarisations.

        Outputs:
        - [Signal]: received symbols as a Signal, shape [R,B,S,2]
        """
        # Filter
        samples = self.filter(samples)

        # Downsample to symbol space
        symbols = Signal(
            samples = samples.samples_time[..., ::self.downsample_factor, :],
            sample_rate = self.filter.symbol_rate
        )

        # Scale back to transmission power after downsampling -> scale to unit power -> multiply by 2 for dual-polarisation transmission
        symbols.samples_time /= np.sqrt(self.downsample_factor * self.power_W / 2) # Maintain unit power after downsampling

        return symbols

    @property
    def filter(self) -> pulse.Pulse:
        """
        [pulse.Pulse] the filter to use for antialiasing.
        """
        return self._filter

    @filter.setter
    def filter(self, value):
        if isinstance(value, pulse.Pulse): self._filter = value
        elif isinstance(value, (list, tuple)): self._filter = getattr(pulse, value[0].upper())(*value[1])
        else: raise ValueError(f"Receiver filter must be a Pulse or str, but had type {type(value)}")

    @property
    def power_dBm(self) -> float:
        """
        [float] the transmission signal power in dBm.
        """
        return self._power_dBm

    @power_dBm.setter
    def power_dBm(self, value: float):
        self._power_dBm = value

    @property
    def power_W(self) -> float:
        """
        [float] the transmission signal power in W.
        """
        return 0.001 * 10 ** (self.power_dBm / 10)

    @power_W.setter
    def power_W(self, value):
        assert value > 0, f"Power in W must be positive, but was {value}"
        self.power_dBm = 10 * np.log10(1000 * value)

    @property
    def downsample_factor(self) -> int:
        """
        [int] the transmitted samples per symbol
        """
        return self._downsample_factor

    @downsample_factor.setter
    def downsample_factor(self, value):
        assert isinstance(value, int), f"downsample_factor must be an integer, but had type {type(value)}"
        assert value > 0, f"downsample_factor must be positive (>0), but was {value}"
        self._downsample_factor = value
