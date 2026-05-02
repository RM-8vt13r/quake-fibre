"""
Test correctness of filter.py
"""
from configparser import ConfigParser
import sys

import numpy as np
try:
    import cupy as cp
except:
    print("cupy not available; skipping all CUDA tests..")

from quakefibre import Signal, Domain, Device, Filter

parameters = ConfigParser()
parameters["TRANSCEIVER"] = {
    "constellation": "QPSK",  # The symbol constellation to use
    "power": "2",             # Transmission power in dBm
    "symbol_rate": "1e6",     # Symbol rate in symbols / s
    "pulse": "RRCOS",         # Pulseshape, can be SINC or RRCOS, or define your own using the Pulse class
    "pulse_parameter": "0.5", # Parameter to pass to the pulse constructor. For a RRCOS pulse, this is the rolloff factor
    "sample_factor": "4"      # Samples per symbol
}

parameters["SIGNAL"] = {
    "symbol_count": "1e2"
}

def test_filter():
    signal = Signal(
            samples = np.array([1, 10, 0.5]),
            sample_rate = 1,
            sample_axis = -1,
            carrier_wavelength = 1550
        )

    filter_zeros = Filter(
            frequencies = signal.frequency % signal.bandwidth,
            responses = np.array([0, 0, 0])
        )

    filter_ones = Filter(
            frequencies = signal.frequency % signal.bandwidth,
            responses = np.array([1, 1, 1])
        )

    filter_skew = Filter(
            frequencies = signal.frequency % signal.bandwidth,
            responses = np.array([0, 1, 2])
        )

    signal_zeros = filter_zeros(signal)
    signal_ones  = filter_ones(signal)
    signal_skew  = filter_skew(signal)

    assert np.allclose(signal_zeros.samples_time, 0), f"Filtered time-domain signal should've been full of zeros, but was {signal.samples_time}"
    assert np.allclose(signal_ones.samples_time, signal.samples_time), f"Filtered time-domain signal should've matched the input signal, but didn't"
    assert not np.allclose(signal_skew.samples_time, signal_zeros.samples_time) and not np.allclose(signal_skew.samples_time, signal_ones.samples_time), f"Filtered time-domain signal shouldn't have matched other signals, but did"

    if 'cupy' not in sys.modules: return

    signal.to_device(Device.CUDA)

    signal_zeros_cuda = filter_zeros(signal)
    signal_ones_cuda  = filter_ones(signal)
    signal_skew_cuda  = filter_skew(signal)

    assert signal_zeros_cuda.device == signal.device, f"Filter input signal was on CUDA, but output signal wasn't"
    assert signal_ones_cuda.device == signal.device, f"Filter input signal was on CUDA, but output signal wasn't"
    assert signal_skew_cuda.device == signal.device, f"Filter input signal was on CUDA, but output signal wasn't"
    
    signal_zeros_cuda.to_device(Device.CPU)
    signal_ones_cuda.to_device(Device.CPU)
    signal_skew_cuda.to_device(Device.CPU)

    assert np.allclose(signal_zeros.samples_time, signal_zeros_cuda.samples_time), "'Zeros'-filtered signal doesn't match between CPU and CUDA"
    assert np.allclose(signal_ones.samples_time, signal_ones_cuda.samples_time), "'Zeros'-filtered signal doesn't match between CPU and CUDA"
    assert np.allclose(signal_skew.samples_time, signal_skew_cuda.samples_time), "'Zeros'-filtered signal doesn't match between CPU and CUDA"
