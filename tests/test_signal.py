"""
Test correctness of signal.py
"""
from configparser import ConfigParser

import numpy as np

from tremor_waveplate_toolbox import Transmitter, Domain

parameters = ConfigParser()
parameters["TRANSCEIVER"] = {
    "constellation": "QPSK",  # The symbol constellation to use
    "power": "2",             # Transmission power in dBm
    "baud_rate": "1e6",       # Baud rate in symbols / s
    "pulse": "RRCOS",         # Pulseshape, can be SINC or RRCOS, or define your own using the Pulse class
    "pulse_parameter": "0.5", # Parameter to pass to the pulse constructor. For a RRCOS pulse, this is the rolloff factor
    "upsample_factor": "4"    # Samples per symbol
}

parameters["SIGNAL"] = {
    "symbol_count": "1e2"
}

def test_signal():
    transmitter = Transmitter(parameters)
    _, signal = transmitter.transmit_random(1, int(parameters.getfloat("SIGNAL", "symbol_count")))

    samples_time1 = signal.samples_time.copy()
    power_time = signal.power_W
    samples_frequency = signal.samples_frequency.copy()
    power_frequency = signal.power_W
    assert signal.domain == Domain.FREQUENCY, f"Signal did not switch to frequency domain properly"
    assert not np.allclose(samples_time1, samples_frequency), f"Signal did not change samples on switch to frequency domain"
    assert np.isclose(power_time, power_frequency), f"Signal power in the time- and frequncy domains does not match"
    samples_time2 = signal.samples_time.copy()
    assert signal.domain == Domain.TIME, f"Signal did not switch to time domain properly"
    assert not np.allclose(samples_time2, samples_frequency), f"Signal did not change samples on switch to time domain"
    assert np.allclose(samples_time1, samples_time2), f"Signal time-domain samples do not match samples before FFT pair"

    signal2 = signal.copy()
    assert signal == signal2, f"Signal copying was not successful"

    signal2.resample(signal2.sample_rate * 10)
    assert signal2.shape == signal.shape[:-2] + (signal.shape[-2] * 10, signal.shape[-1]), f"Expected signal length {signal2.shape[-2] * 10} after resampling, but got {signal.shape[-2]}"
    
    signal2.resample(signal.sample_rate)
    assert signal2.shape == signal.shape, f"Expected signal length {signal.shape[-2]} after double resampling, but got {signal2.shape[-2]}"
    assert np.allclose(signal2.samples_time, signal.samples_time), f"Double resampled signal does not match original signal"