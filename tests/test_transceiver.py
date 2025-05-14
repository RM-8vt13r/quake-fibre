"""
Test correctness of transmitter.py and receiver.py
"""

from configparser import ConfigParser

import numpy as np

from tremor_waveplate_toolbox import Signal, Transmitter, Receiver, QPSK

parameters = ConfigParser()
parameters['TRANSCEIVER'] = {
    'constellation': 'QPSK',
    'power': '2',
    'baud_rate': '1e6',
    'pulse': 'RRCOS',
    'pulse_parameter': '0.7',
    'filter': 'RRCOS',
    'filter_parameter': '0.7',
    'upsample_factor': '10'
}

parameters['SIGNAL'] = {
    'batch_size': '10',
    'symbol_count': '1e3'
}

def test_transmitter():
    transmitter = Transmitter(parameters)
    symbols, signal = transmitter.transmit_random((parameters.getint('SIGNAL', 'batch_size'), int(parameters.getfloat('SIGNAL', 'symbol_count'))))
    assert symbols.shape == (1, parameters.getint('SIGNAL', 'batch_size'), int(parameters.getfloat('SIGNAL', 'symbol_count')), 2), f"Transmitted symbols shape should be [R (1), B ({parameters.getint('SIGNAL', 'batch_size')}), S ({int(parameters.getfloat('SIGNAL', 'symbol_count'))}), P (2)], but was {symbols.shape}"
    assert signal.shape == (1, parameters.getint('SIGNAL', 'batch_size'), int(parameters.getfloat('SIGNAL', 'symbol_count')) * parameters.getint('TRANSCEIVER', 'upsample_factor'), 2), f"Transmitted signal shape should be [R (1), B ({parameters.getint('SIGNAL', 'batch_size')}), S ({int(parameters.getfloat('SIGNAL', 'symbol_count')) * parameters.getint('TRANSCEIVER', 'upsample_factor')}), P (2)], but was {signal.shape}"
    assert np.allclose(symbols.power_W, 2), f"Transmitted symbols should have power 2 (unit power per polarisation), but this was {symbols.power_W}W"
    assert np.allclose(signal.power_dBm, parameters.getfloat('TRANSCEIVER', 'power')), f"Transmitted signal should have a power of {parameters.getfloat('TRANSCEIVER', 'power')}dBm, but this was {signal.power_dBm}dBm"

def test_receiver():
    transmitter = Transmitter(parameters)
    symbols, signal = transmitter.transmit_random((parameters.getint('SIGNAL', 'batch_size'), int(parameters.getfloat('SIGNAL', 'symbol_count'))))

    receiver = Receiver(parameters)
    symbols_received = receiver(signal)
    assert symbols_received.shape == symbols.shape, f"Transmitted and received symbols should have the same shape, but had shapes {symbols.shape} and {symbols_received.shape}"
    assert np.allclose(symbols_received.power_W, 2), f"Received symbols should have power 2 (unit power per polarisation), but had power {received_symbols.power_W}W"
    assert np.allclose(symbols_received.samples_time, symbols.samples_time), f"Transmitted and received symbols should be the same, but were not"

    parameters['TRANSCEIVER']['filter_parameter'] = '0.1'
    receiver_2 = Receiver(parameters)
    symbols_received_2 = receiver_2(signal)
    assert np.all(symbols_received_2.power_W <= 2), f"Received symbols should have power below 2 (unit power per polarisation), but had power {symbols_received_2.power_W}"
    assert not np.allclose(symbols_received_2.samples_time, symbols.samples_time), f"Received symbols with non-matched filter should not be the same as the transmitted symbols, but were"
