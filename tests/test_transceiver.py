"""
Test correctness of transmitter.py and receiver.py
"""

from configparser import ConfigParser

import numpy as np

from tremor_waveplate_toolbox import Signal, Transmitter, QPSK

parameters = ConfigParser()
parameters["TRANSCEIVER"] = {
    "constellation": "QPSK",
    "power": "2",
    "baud_rate": "1e6",
    "pulse": "RRCOS",
    "pulse_parameter": "0.7",
    "upsample_factor": "10"
}

def test_transmitter():
    batch_size = 10
    Nsymb = int(1e5)

    transmitter = Transmitter(parameters)
    symbols, signal = transmitter.transmit_random(batch_size, Nsymb)
    assert symbols.shape == (batch_size, Nsymb, 2), f"Transmitted symbols shape should be [B ({batch_size}), S ({Nsymb}), P (2)], but was {symbols.shape}"
    assert signal.shape == (batch_size, Nsymb * parameters.getint("TRANSCEIVER", "upsample_factor"), 2), f"Transmitted signal shape should be [B ({batch_size}), S ({Nsymb * parameters.getint("TRANSCEIVER", "upsample_factor")}), P (2)], but was {signal.shape}"
    assert np.allclose(symbols.power_W, 2), f"Transmitted symbols should have power 2 (unit power per polarisation), but this was {symbols.power_W}W"
    assert np.allclose(signal.power_dBm, parameters.getfloat("TRANSCEIVER", "power")), f"Transmitted signal should have a power of {parameters.getfloat("TRANSCEIVER", "power")}dBm, but this was {signal.power_dBm}dBm"

def test_receiver():
    pass

def test_transceiver():
    pass
