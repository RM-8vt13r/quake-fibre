"""
Test correctness of pulse.py
"""

import numpy as np
import matplotlib.pyplot as plt

from tremor_waveplate_toolbox import Sinc, RootRaisedCosine, Signal

symbol_rate = 1e6
upsample_factor = 17
sample_rate = symbol_rate * upsample_factor
Nsymb = int(1e5)

def test_normalisation():
    t = (np.arange(Nsymb * upsample_factor) - (Nsymb * upsample_factor) / 2) / sample_rate
    f = np.fft.fftfreq(
        n = len(t),
        d = 1 / sample_rate
    )

    p = Sinc(symbol_rate)
    sinc_t = p.pulse_t(t)
    sinc_f = p.pulse_f(f)
    assert np.isclose(np.sum(np.abs(sinc_t) ** 2) * np.diff(t)[0], 1, rtol = 0, atol = 0.1), f"Sinc pulse did not have unit energy in the time domain"
    assert np.isclose(np.sum(np.abs(sinc_f) ** 2) * np.diff(f)[0], 1, rtol = 0, atol = 0.1) ,f"Sinc pulse did not have unit energy in the frequency domain"

    p = RootRaisedCosine(symbol_rate, 0)
    # rrcos_t = p.pulse_t(t)
    rrcos_f = p.pulse_f(f)
    # assert np.allclose(sinc_t, rrcos_t), f"0-rolloff RRCOS pulse did not match sinc pulse in the time domain"
    assert np.allclose(sinc_f, rrcos_f), f"0-rolloff RRCOS pulse did not match sinc pulse in the frequency domain"

    p = RootRaisedCosine(symbol_rate, 0.5)
    rrcos_f = p.pulse_f(f)
    assert np.isclose(np.sum(np.abs(rrcos_f) ** 2) * np.diff(f)[0], 1, rtol = 0, atol = 0.1), f"Root-raised cosine pulse with rolloff 0.5 did not have unit energy in the frequency domain"

def test_modulation():
    t = np.arange(Nsymb * upsample_factor) / sample_rate
    samples = np.zeros((len(t), 2), dtype = float)
    samples[::3 * upsample_factor, :] = np.sqrt(upsample_factor / 2) # Upsampled pulse train over two polarisations (* upsample_factor to keep power constant, / 2 for dual-polarisation)
    samples[upsample_factor::3 * upsample_factor, :] = np.sqrt(upsample_factor / 2)
    samples[2 * upsample_factor::3 * upsample_factor, :] = -np.sqrt(upsample_factor / 2)

    signal = Signal(
        samples = samples,
        sample_rate = sample_rate
    )
    assert np.isclose(signal.energy, len(t)), f"Upsampled signal does not have unit power"

    sinc = Sinc(symbol_rate) # Unit energy pulseshape
    sinc_signal = sinc(signal) # Modulate
    assert np.isclose(sinc_signal.power_W, 1, atol = 0, rtol = 0.1), f"Sinc-modulated optical signal does not have unit power"

    rrcos = RootRaisedCosine(symbol_rate, 0.5) # Unit energy pulseshape
    rrcos_signal = rrcos(signal) # Modulate
    assert np.isclose(rrcos_signal.power_W, 1, atol = 0, rtol = 0.1), f"RRCOS-modulated optical signal does not have unit power"
