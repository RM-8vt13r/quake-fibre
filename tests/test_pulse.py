"""
Test correctness of pulse.py
"""

import numpy as np
import matplotlib.pyplot as plt

from tremor_waveplate_toolbox import Sinc, RootRaisedCosine, Square, Signal

symbol_rate = 1e6
upsample_factor = 17
sample_rate = symbol_rate * upsample_factor
symbol_count = int(1e5)

def test_normalisation():
    time = (np.arange(symbol_count * upsample_factor) - (symbol_count * upsample_factor) / 2) / sample_rate
    frequency = np.fft.fftfreq(
        n = len(time),
        d = 1 / sample_rate
    )

    pulse = Sinc(symbol_rate)
    sinc_time = pulse.pulse_time(time)
    sinc_frequency = pulse.pulse_frequency(frequency)
    assert np.isclose(np.sum(np.abs(sinc_time) ** 2) * np.diff(time)[0], 1, rtol = 0, atol = 0.1), f"Sinc pulse did not have unit energy in the time domain"
    assert np.isclose(np.sum(np.abs(sinc_frequency) ** 2) * np.diff(frequency)[0], 1, rtol = 0, atol = 0.1) ,f"Sinc pulse did not have unit energy in the frequency domain"

    pulse = RootRaisedCosine(symbol_rate, 0)
    # rrcos_t = p.pulse_t(t)
    rrcos_frequency = pulse.pulse_frequency(frequency)
    # assert np.allclose(sinc_t, rrcos_t), f"0-rolloff RRCOS pulse did not match sinc pulse in the time domain"
    assert np.allclose(sinc_frequency, rrcos_frequency), f"0-rolloff RRCOS pulse did not match sinc pulse in the frequency domain"

    pulse = RootRaisedCosine(symbol_rate, 0.5)
    rrcos_frequency = pulse.pulse_frequency(frequency)
    assert np.isclose(np.sum(np.abs(rrcos_frequency) ** 2) * np.diff(frequency)[0], 1, rtol = 0, atol = 0.1), f"Root-raised cosine pulse with rolloff 0.5 did not have unit energy in the frequency domain"

    pulse = Square(symbol_rate)
    square_time = pulse.pulse_time(time)
    square_frequency = pulse.pulse_frequency(frequency)
    assert np.isclose(np.sum(np.abs(square_time) ** 2) * np.diff(time)[0], 1, rtol = 0, atol = 0.1), f"Square pulse did not have unit energy in the time domain"
    assert np.isclose(np.sum(np.abs(square_frequency) ** 2) * np.diff(frequency)[0], 1, rtol = 0, atol = 0.1), f"Square pulse did not have unit energy in the frequency domain"

def test_modulation():
    time = np.arange(symbol_count * upsample_factor) / sample_rate
    samples = np.zeros((1, 1, len(time), 2), dtype = float)
    samples[0, 0, ::3 * upsample_factor, :] = np.sqrt(upsample_factor / 2) # Upsampled pulse train over two polarisations (* upsample_factor to keep power constant, / 2 for dual-polarisation)
    samples[0, 0, upsample_factor::3 * upsample_factor, :] = np.sqrt(upsample_factor / 2)
    samples[0, 0, 2 * upsample_factor::3 * upsample_factor, :] = -np.sqrt(upsample_factor / 2)

    signal = Signal(
        samples = samples,
        sample_rate = sample_rate
    )
    assert np.isclose(signal.energy, len(time)), f"Upsampled signal does not have unit power"

    pulse_sinc = Sinc(symbol_rate) # Unit energy pulseshape
    signal_sinc = pulse_sinc(signal) # Modulate
    assert np.isclose(signal_sinc.power_W, 1, atol = 0, rtol = 0.1), f"Sinc-modulated optical signal does not have unit power"

    pulse_rrcos = RootRaisedCosine(symbol_rate, 0.5) # Unit energy pulseshape
    signal_rrcos = pulse_rrcos(signal) # Modulate
    assert np.isclose(signal_rrcos.power_W, 1, atol = 0, rtol = 0.1), f"RRCOS-modulated optical signal does not have unit power"

    pulse_square = Square(symbol_rate)
    signal_square = pulse_square(signal)
    assert np.isclose(signal_square.power_W, 1, atol = 0, rtol = 0.1), f"Square-modulated optical signal does not have unit power"