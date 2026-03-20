"""
A class to transmit optical signals
"""

from configparser import ConfigParser
import json
import logging

import numpy as np

from .signal import Signal
from .constants import Gain
from .utilities import dB2linear, linear2dB
from . import constellation as const
from . import pulse

logger = logging.getLogger()

class Transceiver:
    """
    A class that acts as a transmitter and receiver.
    The transmitter side (randomly) samples symbols, upsamples and modulates them, or transmits an unmodulated continuous wave.
    The receiver side downsamples, first applying an anti-aliasing filter if receiving a modulated signal.
    """
    def __init__(self, parameters: ConfigParser):
        """
        Create a new Transmitter.

        Required and optional entries in parameters['TRANSCEIVER']:
        - constellation [str]: 'BPSK', 'PSK8', 'QPSK', 'QAM4', 'QAM16', 'QAM64', or a string with a list of complex symbols
        - power [float]: Transmission power in dBm
        - symbol_rate [float]: symbol rate in Hz
        - pulse [str]: (optional) 'SINC' or 'RRCOS' for sinc or root-raised cosine pulses
        - pulse_parameter [object]: (optional) any parameter(s) to define the pulse, such as rolloff factor
        - filter [str]: (optional) 'SINC' or 'RRCOS' for sinc or root-raised cosine antialiasing filters
        - filter_parameter [object]: (optional) any parameter(s) to define the filter, such as rolloff factor
        - sample_factor [int]: samples per symbol
        """
        if 'constellation' in parameters['TRANSCEIVER']:
            try:
                self.constellation = json.loads(parameters.get('TRANSCEIVER', 'constellation'))
            except:
                self.constellation = getattr(const, parameters.get('TRANSCEIVER', 'constellation'))
        else:
            self.constellation = None

        self._pulse  = None
        self._filter = None
        self.symbol_rate = parameters.getfloat('TRANSCEIVER', 'symbol_rate')

        self.pulse   = self._construct_pulse('pulse', parameters)
        self.filter  = self._construct_pulse('filter', parameters)
        if self.filter is None and self.pulse is not None:
            self.filter = self.pulse

        self.power_dBm     = parameters.getfloat('TRANSCEIVER', 'power')
        self.sample_factor = int(parameters.getfloat('TRANSCEIVER', 'sample_factor'))

    def _construct_pulse(self, pulse_name: str, parameters: ConfigParser):
        if pulse_name in parameters['TRANSCEIVER']:
            constructor = getattr(pulse, parameters.get('TRANSCEIVER', pulse_name).upper())
            constructor_arguments = [parameters.getfloat('TRANSCEIVER', 'symbol_rate')]
            if f'{pulse_name}_parameter' in parameters['TRANSCEIVER']:
                constructor_arguments.append(parameters.getfloat('TRANSCEIVER', f'{pulse_name}_parameter'))
            
            return constructor(*constructor_arguments)
            
        return None
        
    def transmit_continuous(self, symbol: np.ndarray, symbol_count: int, carrier_wavelength: float = 1550.) -> Signal:
        """
        Create a continuous-wave Signal by repeating the same sample over and over

        Inputs:
        - symbol [np.ndarray]: the symbol to repeat, shape [P] where P = 2 indexes two orthogonal polarisations.
        - symbol_count [int]: the number of (identicals) symbols to transmit. Each symbol is upsampled to self.sample_factor identical samples
        - carrier_wavelength [int]: the signal carrier wavelength in nm

        Outputs:
        - [Signal]: the output signal, shape [R,B,S,P] where R = 1 is fibre realisations, B = 1 is batches and S = sample_count
        """
        assert isinstance(symbol, (tuple, list, np.ndarray)), f"symbol must be a np.ndarray, but was a {type(symbol)}"
        symbol = np.array(symbol, dtype = complex)
        assert len(symbol.shape) == 1 and symbol.shape[0] == 2, f"symbol must have shape [2,], but had shape {symbol.shape}"
        assert isinstance(symbol_count, (int, np.integer)), f"symbol_count must be an int, but was a {type(symbol_count)}"
        assert symbol_count > 0, f"symbol_count must be > 0, but was {symbol_count}"

        # Normalise symbol to unit energy
        symbol /= np.linalg.norm(symbol)

        # Generate continuous-wave signal
        signal = Signal(
            samples = np.full(shape = (1, 1, symbol_count * self.sample_factor, 2), fill_value = symbol, dtype = complex),
            sample_rate = self.symbol_rate * self.sample_factor,
            carrier_wavelength = carrier_wavelength
        )

        # Scale to transmission power
        signal.samples_time *= np.sqrt(self.power_W)

        return signal

    def transmit_symbols(self, symbols: np.ndarray, carrier_wavelength: float = 1550.) -> Signal:
        """
        Pulseshape and amplify a given sequence of symbols

        Inputs:
        - symbols [np.ndarray]: the unit-power sequence of symbols to transmit, shape [B,S,P] where B is batch size, S is the sequence length, and P = 2 indexes two orthogonal polarisations.
        - carrier_wavelength [float]: the signal carrier wavelength in nm

        Outputs:
        - [Signal]: modulated symbols as a Signal, shape [R,B,S,P]
        - [Signal]: the output signal, shape [R,B,U*S,P] where R = 1 is fibre realisations and U is the upsample factor (samples per symbol).
        """
        assert self.pulse is not None, f"Cannot transmit symbols if no pulseshape was defined"
        assert len(symbols.shape) == 3 and symbols.shape[-1] == 2, f"symbols should have shape [B,S,P] with P = 2, but had shape {symbols.shape}"

        symbols = Signal(
            samples = symbols[None].astype(complex),
            sample_rate = self.symbol_rate
        )

        # Upsample to sample space
        samples = Signal(
            samples = np.zeros([*symbols.shape[:-2], self.sample_factor * symbols.shape[-2], symbols.shape[-1]], dtype = complex),
            sample_rate = self.sample_factor * self.symbol_rate,
            carrier_wavelength = carrier_wavelength
        )
        samples.samples_time[..., ::self.sample_factor, :] = symbols.samples_time

        # Scale back to unit power after upsampling -> scale to transmission power -> divide by 2 for dual-polarisation transmission
        samples.samples_time *= np.sqrt(self.sample_factor * self.power_W / 2) # Maintain unit power after upsampling

        # Pulseshape
        samples = self.pulse(samples)

        return symbols, samples

    def transmit_random_symbols(self, batch_size: int, symbol_count: int, carrier_wavelength: float = 1550.) -> Signal:
        """
        Generate a random pulseshaped sequence.

        Inputs:
        - batch_size: the signal batch size B
        - symbol_count: the number of symbols S per batch sample
        - carrier_wavelength [float]: the signal carrier wavelength in nm

        Outputs:
        - [Signal]: modulated symbols as a Signal, shape [R,B,S,P] with number of fibre realisations R = 1 and number of orthogonal polarisations P = 2
        - [Signal]: the output signal, shape [R,B,U*S,2]
        """
        assert self.constellation is not None, f"Cannot draw symbols if constellation was defined"

        # Sample symbol array from constellation
        symbols = self.constellation((batch_size, symbol_count, 2))
        return self.transmit_symbols(symbols, carrier_wavelength)

    def receive_continuous(self, samples: Signal):
        """
        Attenuate and downsample, but don't anti-aliasing filter, a given sequence of samples

        Inputs:
        - samples [Signal]: the sequence of samples to receive, shape [R,B,S,P] where R is the number of realisations, B is batch size, S is the sequence length, and P = 2 indexes two orthogonal polarisations.

        Outputs:
        - [Signal]: received symbols as a Signal, shape [R,B,S,P]
        """
        # Downsample to symbol space
        symbols = Signal(
            samples = samples.samples_time[..., ::self.sample_factor, :].copy(),
            sample_rate = self.symbol_rate
        )

        # Scale back to transmission power after downsampling -> scale to unit power -> multiply by 2 for dual-polarisation transmission
        symbols.samples_time /= np.sqrt(self.power_W) # Maintain unit power after downsampling

        return symbols

    def receive_symbols(self, samples: Signal):
        """
        Attenuate, anti-aliasing filter, and downsample a given sequence of samples

        Inputs:
        - samples [Signal]: the sequence of samples to receive, shape [R,B,S,P] where R is the number of realisations, B is batch size, S is the sequence length, and P = 2 indexes two orthogonal polarisations.

        Outputs:
        - [Signal]: received symbols as a Signal, shape [R,B,S,P]
        """
        assert self.filter is not None, f"Cannot receive symbols when no antialiasing filter was defined"
        samples = self.filter(samples)
        samples.samples /= np.sqrt(self.sample_factor / 2)
        return self.receive_continuous(samples)

    @property
    def constellation(self) -> const.Constellation:
        """
        [const.Constellation] the constellation to transmit symbols from.
        """
        return self._constellation

    @constellation.setter
    def constellation(self, value: const.Constellation):
        if value is None:
            self._constellation = None
            return

        if isinstance(value, (const.Constellation)):
            self._constellation = value
            return
        
        if isinstance(value, (list, tuple, np.ndarray)):
            self._constellation = const.Constellation(value)
            return

        raise ValueError(f"Transmitter constellation must be a Constellation, or list, but had type {type(value)}")

    @property
    def pulse(self):
        """
        [pulse.Pulse] the pulse to use for pulse amplitude modulation.
        """
        return self._pulse

    @pulse.setter
    def pulse(self, value):
        assert value is None or isinstance(value, pulse.Pulse), f"pulse must be a Pulse, but was a {type(value)}"

        if value is not None and value.symbol_rate != self.symbol_rate:
            logger.warning(f"New transceiver pulse implicitely changed symbol rate from {self.symbol_rate} to {value.symbol_rate} symbols per second")
            self.symbol_rate = value.symbol_rate

        self._pulse = value

    @property
    def filter(self):
        """
        [pulse.Pulse] the filter to use for antialiasing.
        """
        return self._filter

    @filter.setter
    def filter(self, value):
        assert value is None or isinstance(value, pulse.Pulse), f"filter must be a Pulse, but was a {type(value)}"

        if value is not None and value.symbol_rate != self.symbol_rate:
            logger.warning(f"New transceiver filter implicitely changed symbol rate from {self.symbol_rate} to {value.symbol_rate} symbols per second")
            self.symbol_rate = value.symbol_rate

        self._filter = value

    @property
    def symbol_rate(self):
        """
        [float] The rate with which symbols are transmitted in symbols per second.
        """
        if self.pulse is not None:
            return self.pulse.symbol_rate

        if self.filter is not None:
            return self.filter.symbol_rate

        return self._symbol_rate

    @symbol_rate.setter
    def symbol_rate(self, value):
        assert isinstance(value, (int, float, np.integer, np.floating)), f"symbol_rate must be a float, but was a {type(value)}"
        assert value > 0, f"symbol_rate must be >0, but was {value}"
        
        if self.pulse is not None:
            self.pulse.symbol_rate = value

        if self.filter is not None:
            self.filter.symbol_rate = value

        self._symbol_rate = value

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
        return 0.001 * dB2linear(self.power_dBm, Gain.POWER)

    @power_W.setter
    def power_W(self, value):
        assert value > 0, f"Power in W must be positive, but was {value}"
        self.power_dBm = linear2dB(1000 * value, Gain.POWER)

    @property
    def sample_factor(self) -> int:
        """
        [int] the transmitted samples per symbol
        """
        return self._sample_factor

    @sample_factor.setter
    def sample_factor(self, value):
        assert isinstance(value, (int, np.integer)), f"sample_factor must be an integer, but had type {type(value)}"
        assert value > 0, f"sample_factor must be >0, but was {value}"
        self._sample_factor = value
