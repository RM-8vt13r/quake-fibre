"""
Test correctness of signal.py
"""
from configparser import ConfigParser
import sys

import numpy as np
try:
    import cupy as cp
except:
    print("cupy not available; skipping all CUDA tests..")

from quakefibre import Transceiver, Domain, Device

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

def test_signal():
    transceiver = Transceiver(parameters)
    _, signal = transceiver.transmit_random_symbols(1, int(parameters.getfloat("SIGNAL", "symbol_count")))

    for device in (Device.CPU, Device.CUDA):
        if device == Device.CUDA and 'cupy' not in sys.modules: return

        signal.to_device(device)

        samples_time1 = signal.samples_time.copy()
        power_time = signal.power_W
        samples_frequency = signal.samples_frequency.copy()
        power_frequency = signal.power_W
        assert signal.domain == Domain.FREQUENCY, f"{device} signal did not switch to frequency domain properly"
        assert not signal.xp.allclose(samples_time1, samples_frequency), f"{device} signal did not change samples on switch to frequency domain"
        assert signal.xp.isclose(power_time, power_frequency), f"{device} signal power in the time- and frequncy domains does not match"
        samples_time2 = signal.samples_time.copy()
        assert signal.domain == Domain.TIME, f"{device} signal did not switch to time domain properly"
        assert not signal.xp.allclose(samples_time2, samples_frequency), f"{device} signal did not change samples on switch to time domain"
        assert signal.xp.allclose(samples_time1, samples_time2), f"{device} signal time-domain samples do not match samples before FFT pair"
        assert signal.xp.allclose(signal.xp.diff(signal.xp.fft.fftshift(signal.frequency)), signal.sample_bandwidth), f"Signal sample_bandwidth ({signal.sample_bandwidth}Hz) does not match frequency sample spacing ({np.diff(signal.xp.fft.fftfreq(signal.frequency))})"
        assert signal.xp.isclose(signal.xp.fft.fftshift(signal.frequency)[-1] - signal.xp.fft.fftshift(signal.frequency)[0] + signal.sample_bandwidth, signal.bandwidth), f"Signal frequency samples cover {signal.xp.fft.fftshift(signal.frequency)[-1] - signal.xp.fft.fftshift(signal.frequency)[0] + signal.sample_bandwidth}Hz of bandwidth, but signal.bandwidth returns {signal.bandwidth}Hz"

        signal2 = signal.copy()
        assert signal == signal2, f"{device} signal copying was not successful"

        signal2.resample(signal2.sample_rate * 10)
        assert signal2.shape == signal.shape[:-2] + (signal.shape[-2] * 10, signal.shape[-1]), f"{device} expected signal length {signal2.shape[-2] * 10} after resampling, but got {signal.shape[-2]}"
        
        signal2.resample(signal.sample_rate)
        assert signal2.shape == signal.shape, f"{device} expected signal length {signal.shape[-2]} after double resampling, but got {signal2.shape[-2]}"
        assert signal.xp.allclose(signal2.samples_time, signal.samples_time), f"{device} double resampled signal does not match original signal"

    signal.to_device(Device.CPU)
    signal_cuda = signal.copy()
    signal_cuda.to_device(Device.CUDA)

    assert signal == signal_cuda, f"Copying between CPU and CUDA changed the Signal"

    assert np.allclose(signal.samples_time, signal_cuda.samples_time.get()), f"CUDA signal time-domain samples don't match CPU signal"
    assert np.allclose(signal.samples_frequency, signal_cuda.samples_frequency.get()), f"CUDA signal frequency-domain samples don't match CPU signal"