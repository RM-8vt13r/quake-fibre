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

parameters["SIGNAL"] = {
    "batch_size": "10",
    "symbol_count": "1e3"
}

def test_transmitter():
    transmitter = Transmitter(parameters)
    symbols, signal = transmitter.transmit_random((parameters.getint("SIGNAL", "batch_size"), int(parameters.getfloat("SIGNAL", "symbol_count"))))
    assert symbols.shape == (1, parameters.getint("SIGNAL", "batch_size"), int(parameters.getfloat("SIGNAL", "symbol_count")), 2), f"Transmitted symbols shape should be [R (1), B ({parameters.getint("SIGNAL", "batch_size")}), S ({int(parameters.getfloat("SIGNAL", "symbol_count"))}), P (2)], but was {symbols.shape}"
    assert signal.shape == (1, parameters.getint("SIGNAL", "batch_size"), int(parameters.getfloat("SIGNAL", "symbol_count")) * parameters.getint("TRANSCEIVER", "upsample_factor"), 2), f"Transmitted signal shape should be [R (1), B ({parameters.getint("SIGNAL", "batch_size")}), S ({int(parameters.getfloat("SIGNAL", "symbol_count")) * parameters.getint("TRANSCEIVER", "upsample_factor")}), P (2)], but was {signal.shape}"
    assert np.allclose(symbols.power_W, 2), f"Transmitted symbols should have power 2 (unit power per polarisation), but this was {symbols.power_W}W"
    assert np.allclose(signal.power_dBm, parameters.getfloat("TRANSCEIVER", "power")), f"Transmitted signal should have a power of {parameters.getfloat("TRANSCEIVER", "power")}dBm, but this was {signal.power_dBm}dBm"

def test_receiver():
    pass

def test_transceiver():
    pass
