"""
A class to transmit- and receive optical signals
"""

from configparser import ConfigParser

import numpy as np

from .signal import Signal
from . import constellation as const
from . import pulse

class Transmitter:
    """
    A class that acts as a transmitter.
    It randomly samples symbols, upsamples and modulates them.
    """
    def __init__(self, parameters: ConfigParser):
        """
        Create a new Transmitter.

        Required entries in parameters['TRANSCEIVER']:
        - constellation [str, list]: 'BPSK', 'PSK8', 'QPSK', 'QAM4', 'QAM16', 'QAM64', or a list of complex symbols
        - power [float]: Transmission power in dBm
        - baud_rate [float]: symbol rate in Hz
        - pulse [str]: 'SINC' or 'RRCOS' for sinc or root-raised cosine pulses
        - pulse_parameter [object]: any parameter(s) to define the pulse, such as rolloff factor
        - upsample_factor [int]: samples per symbol
        """
        self.constellation   = parameters.get('TRANSCEIVER', 'constellation')
        self.pulse           = [parameters.get('TRANSCEIVER', 'pulse'), (parameters.getfloat('TRANSCEIVER', 'baud_rate'), parameters.getfloat('TRANSCEIVER', 'pulse_parameter'))]
        self.power_dBm       = parameters.getfloat('TRANSCEIVER', 'power')
        self.upsample_factor = int(parameters.getfloat('TRANSCEIVER', 'upsample_factor'))

    def __call__(self, symbols: np.ndarray) -> Signal:
        """
        See transmit_symbols
        """
        return self.transmit_symbols(symbols)

    def transmit_symbols(self, symbols: np.ndarray) -> Signal:
        """
        Pulseshape and amplify a given sequence of symbols

        Inputs:
        - symbols [np.ndarray]: the unit-power sequence of symbols to transmit, shape [B,S,2] where B is batch size, S is the sequence length, and the last dimension indexes two orthogonal polarisations.

        Outputs:
        - [Signal]: modulated symbols as a Signal, shape [R,B,S,2]
        - [Signal]: the output signal, shape [R,B,U*S,2] where R=1 is fibre realisations and U is the upsample factor (samples per symbol).
        """
        symbols = Signal(
            samples = symbols[None],
            sample_rate = self.pulse.symbol_rate
        )

        # Upsample to sample space
        samples = Signal(
            samples = np.zeros([*symbols.shape[:-2], self.upsample_factor * symbols.shape[-2], symbols.shape[-1]]),
            sample_rate = self.upsample_factor * self.pulse.symbol_rate
        )
        samples.samples_time[..., ::self.upsample_factor, :] = symbols.samples_time

        # Scale back to unit power after upsampling -> divide by 2 for dual-polarisation transmission -> scale to transmission power
        samples.samples_time *= np.sqrt(self.upsample_factor * self.power_W / 2) # Maintain unit power after upsampling

        # Pulseshape
        samples = self.pulse(samples)

        return symbols, samples

    def transmit_random(self, shape: tuple) -> Signal:
        """
        Generate a random pulseshaped sequence.

        Inputs:
        - shape [tuple]: the shape [B,S] in which to generate sequences, P = 2 is added at the end.

        Outputs:
        - [Signal]: modulated symbols as a Signal, shape [R,B,S,2]
        - [Signal]: the output signal, shape [R,B,U*S,2]
        """
        # Sample symbol array from constellation
        symbols = self.constellation((*shape, 2))
        return self.transmit_symbols(symbols)

    @property
    def constellation(self) -> const.Constellation:
        """
        [const.Constellation] the constellation to transmit symbols from.
        """
        return self._constellation

    @constellation.setter
    def constellation(self, value: const.Constellation):
        if isinstance(value, (const.Constellation)): self._constellation = value
        elif isinstance(value, (list, tuple)): self._constellation = const.Constellation(value)
        elif isinstance(value, str):  self._constellation = getattr(const, value.upper())
        else: raise ValueError(f"Transmitter constellation must be a Constellation, str, or list, but had type {type(value)}")

    @property
    def pulse(self) -> pulse.Pulse:
        """
        [pulse.Pulse] the pulse to use for pulse amplitude modulation.
        """
        return self._pulse

    @pulse.setter
    def pulse(self, value):
        if isinstance(value, pulse.Pulse): self._pulse = value
        elif isinstance(value, (list, tuple)): self._pulse = getattr(pulse, value[0].upper())(*value[1])
        else: raise ValueError(f"Transmitter pulse must be a Pulse or str, but had type {type(value)}")

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
