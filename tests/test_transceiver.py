"""
Test correctness of transceiver.py
"""

from configparser import ConfigParser

import numpy as np

from tremor_waveplate_toolbox import Signal, Transceiver, QPSK

parameters = ConfigParser()
parameters['TRANSCEIVER'] = {
    'constellation': 'QPSK',
    'power': '2',
    'symbol_rate': '1e6',
    'pulse': 'RRCOS',
    'pulse_parameter': '0.7',
    'filter': 'RRCOS',
    'filter_parameter': '0.7',
    'sample_factor': '10'
}

parameters['SIGNAL'] = {
    'batch_size': '10',
    'symbol_count': '1e3'
}

def test_transmitter():
    transceiver = Transceiver(parameters)
    symbols, signal = transceiver.transmit_random_symbols(parameters.getint('SIGNAL', 'batch_size'), int(parameters.getfloat('SIGNAL', 'symbol_count')))
    assert symbols.shape == (1, parameters.getint('SIGNAL', 'batch_size'), int(parameters.getfloat('SIGNAL', 'symbol_count')), 2), f"Transmitted symbols shape should be [R (1), B ({parameters.getint('SIGNAL', 'batch_size')}), S ({int(parameters.getfloat('SIGNAL', 'symbol_count'))}), P (2)], but was {symbols.shape}"
    assert signal.shape == (1, parameters.getint('SIGNAL', 'batch_size'), int(parameters.getfloat('SIGNAL', 'symbol_count')) * parameters.getint('TRANSCEIVER', 'sample_factor'), 2), f"Transmitted signal shape should be [R (1), B ({parameters.getint('SIGNAL', 'batch_size')}), S ({int(parameters.getfloat('SIGNAL', 'symbol_count')) * parameters.getint('TRANSCEIVER', 'sample_factor')}), P (2)], but was {signal.shape}"
    assert np.allclose(symbols.power_W, 2), f"Transmitted symbols should have power 2 W (unit power per polarisation), but this was {symbols.power_W} W"
    assert np.allclose(signal.power_dBm, parameters.getfloat('TRANSCEIVER', 'power')), f"Transmitted signal should have a power of {parameters.getfloat('TRANSCEIVER', 'power')} dBm ({0.001 * 10 ** (parameters.getfloat('TRANSCEIVER', 'power') / 10)} W), but this was {signal.power_dBm} dBm ({signal.power_W} W)"
    transceiver.constellation = [1, 0]
    symbols, signal = transceiver.transmit_random_symbols(parameters.getint('SIGNAL', 'batch_size'), int(parameters.getfloat('SIGNAL', 'symbol_count')))

    signal = transceiver.transmit_continuous([10, 0], int(parameters.getfloat('SIGNAL', 'symbol_count')), )
    assert signal.shape == (1, 1, int(parameters.getfloat('SIGNAL', 'symbol_count')) * parameters.getint('TRANSCEIVER', 'sample_factor'), 2), f"Continuous-wave signal shape should be [R (1), B (1), S ({int(parameters.getfloat('SIGNAL', 'symbol_count')) * parameters.getint('TRANSCEIVER', 'sample_factor')}), P (2)], but was {signal.shape}"
    assert np.allclose(signal.power_dBm, parameters.getfloat('TRANSCEIVER', 'power')), f"Continuous-wave signal should have a power of {parameters.getfloat('TRANSCEIVER', 'power')} dBm, but this was {signal.power_dBm} dBm"
    
def test_receiver():
    transceiver = Transceiver(parameters)
    symbols, signal = transceiver.transmit_random_symbols(parameters.getint('SIGNAL', 'batch_size'), int(parameters.getfloat('SIGNAL', 'symbol_count')))

    symbols_received = transceiver.receive_symbols(signal)
    assert symbols_received.shape == symbols.shape, f"Transmitted and received symbols should have the same shape, but had shapes {symbols.shape} and {symbols_received.shape}"
    assert np.allclose(symbols_received.power_W, 2), f"Received symbols should have power 2 (unit power per polarisation), but had power {symbols_received.power_W}W"
    assert np.allclose(symbols_received.samples_time, symbols.samples_time), f"Transmitted and received symbols should be the same, but were not"

    parameters['TRANSCEIVER']['filter_parameter'] = '0.1'
    transceiver_2 = Transceiver(parameters)
    symbols_received_2 = transceiver_2.receive_symbols(signal)
    assert np.all(symbols_received_2.power_W <= 2), f"Received symbols should have power below 2 (unit power per polarisation), but had power {symbols_received_2.power_W}"
    assert not np.allclose(symbols_received_2.samples_time, symbols.samples_time), f"Received symbols with non-matched filter should not be the same as the transmitted symbols, but were"

    signal = transceiver.transmit_continuous([10, 0], int(parameters.getfloat('SIGNAL', 'symbol_count')), )
    symbols_received = transceiver.receive_continuous(signal)
    assert symbols_received.shape == (1, 1, int(parameters.getfloat('SIGNAL', 'symbol_count')), 2), f"Received continuous-wave symbols shape should be [R (1), B (1), S ({int(parameters.getfloat('SIGNAL', 'symbol_count'))}), P (2)], but was {signal.shape}"
    assert np.allclose(symbols_received.power_W, 1), f"Received continuous-wave symbols should have a power of 1 W, but this was {symbols_received.power_W} W"
    