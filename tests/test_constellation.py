"""
Test correctness of constellation.py
"""

import numpy as np

from quakefibre import QPSK

def test_normalisation():
    c = QPSK.copy()
    assert np.allclose(c.coordinates, np.array([a + 1j * b for a in (1, -1) for b in (1, -1)]) / np.sqrt(2)), "QPSK coordinates were not scaled correctly for unit mean energy"
    assert np.allclose(c.probabilities, 0.25), "QPSK probabilities were not scaled correctly to add to 1"

    c.coordinates = [2 + 2j, 1 - 1j, -1 + 1j, -1 - 1j] # Average power = (8 + 3 * 2) / 4 = 3.5
    assert np.allclose(c.coordinates, np.array(np.array([2 + 2j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(3.5))), "QPSK coordinates not scaled correctly after coordinate reassignment"

    c.probabilities = [2, 6, 6, 6] # Probabilities = [0.1, 0.3, 0.3, 0.3], average power = 0.1 * 8 + 0.3 * 3 * 2 = 2.6
    assert np.allclose(c.probabilities, np.array([0.1, 0.3, 0.3, 0.3])), f"QPSK probabilities not scaled correctly after probability reassignment"
    assert np.allclose(c.coordinates, np.array([2 + 2j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2.6)), "QPSK coordinates not scaled correctly after probability reassignment"

def test_generation():
    c = QPSK.copy()

    Nsymb = int(1e6)
    sequence = c((Nsymb,))
    assert np.all(np.sum(sequence[:,None] == c.coordinates[None,:], axis = 1) == 1), f"QPSK constellation generates symbols that are not in its constellation"
    assert np.isclose(np.sum(np.abs(sequence) ** 2), Nsymb, rtol = 0.005, atol = 0), f"QPSK generated symbol sequence does not approach unit power"

    c.coordinates = [2 + 2j, 1 - 1j, -1 + 1j, -1 - 1j]
    c.probabilities = [2, 6, 6, 6]
    sequence = c((Nsymb,))
    assert np.all(np.sum(sequence[:,None] == c.coordinates[None,:], axis = 1) == 1), f"Custom constellation generates symbols that are not in its constellation"
    assert np.isclose(np.sum(np.abs(sequence) ** 2), Nsymb, rtol = 0.005, atol = 0), f"Custom generated symbol sequence does not approach unit power"
