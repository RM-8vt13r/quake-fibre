"""
Class representing a finite impulse response filter.
"""
import numpy as np
try:
    import cupy as cp
except:
    pass

from .signal import Signal
from .constants import Domain

class Filter:
    def __init__(self,
            frequencies: np.ndarray,
            responses: np.ndarray,
        ):
        """
        Create a new finite impulse response filter in the frequency domain.
        
        Inputs:
        - frequencies [np.ndarray]: list of frequencies in Hz (see responses), shape [F]
        - responses [np.ndarray]: list of frequency responses, corresponding to frequencies, shape [F]
        """
        assert isinstance(frequencies, (np.ndarray, list, tuple)), f"frequencies must be a np.ndarray, but was a {type(frequencies)}"
        assert isinstance(responses, (np.ndarray, list, tuple)), f"responses must be a np.ndarray, but was a {type(responses)}"
        frequencies = np.array(frequencies)
        responses = np.array(responses)
        assert len(frequencies.shape) == 1, f"frequencies must have shape [F,], but had shape {frequencies.shape}"
        assert len(responses.shape) == 1, f"responses must have shape [F,], but had shape {responses.shape}"
        assert frequencies.shape == responses.shape, f"frequencies and responses must have the same shape, but had shapes {frequencies} and {responses}"
        assert isinstance(frequencies[0], (np.integer, np.floating)), f"frequencies must be real, but had dtype {frequencies.dtype}"
        assert isinstance(responses[0], (np.integer, np.floating)), f"responses must be real, but had dtype {responses.dtype}"
        
        self._frequencies = frequencies
        self._responses = responses

    def __call__(self, signal: Signal):
        """
        See filtered()
        """
        return self.filtered(signal)

    def filtered(self, signal: Signal):
        """
        Filter a signal along its frequency axis. If the filter is not defined at the signal frequencies, interpolate it linearly. This function internally copies the signal.

        Inputs:
        - signal [Signal]: the input signal

        Outputs:
        - [Signal] the output signal
        """
        filter_signal_interpolated = Signal(
            samples = signal.xp.interp(signal.frequency % signal.bandwidth, signal.xp.array(self.frequencies), signal.xp.array(self.responses))[*(None,) * signal.sample_axis_nonnegative, :, *(None,) * (-1 - signal.sample_axis_negative)],
            sample_rate = signal.sample_rate,
            sample_axis = signal.sample_axis,
            domain = Domain.FREQUENCY,
        )
        signal_filtered = signal.copy()
        signal_filtered.samples_frequency = signal_filtered.samples_frequency * filter_signal_interpolated.samples_frequency
        return signal_filtered

    @property
    def frequencies(self):
        """
        [np.ndarray] The frequencies at which the filter response is defined.
        """
        return self._frequencies

    @frequencies.setter
    def frequencies(self, value):
        raise AttributeError("Cannot set frequencies directly; make a new Filter instead")

    @property
    def responses(self):
        """
        [np.ndarray] The frequency responses, corresponding to self.frequencies
        """
        return self._responses

    @responses.setter
    def responses(self, value):
        raise AttributeError("Cannot set responses directly; make a new Filter instead")