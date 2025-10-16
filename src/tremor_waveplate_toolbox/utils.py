"""
Utilities for use in optical fibre modelling.
"""
import numpy as np

from .constants import PAULI_1, PAULI_3

def rotation_matrix(angle: (float, np.ndarray)) -> np.ndarray:
    """
    Create a rotation matrix or matrices from a (set of) angle(s)

    Inputs:
    - angle [float, np.narray]: the angles to transform into a matrix, shape [...] in case of a np.ndarray.

    Outputs:
    - np.ndarray: dtype float, matrix or matrices, shape [..., 2, 2] if angle is a np.ndarray, and [2, 2] if it's a float
    """
    angle = np.array(angle)
    assert angle.dtype in (int, float), f"Angle must have type float, but had type {angle.dtype}"
    angle = angle.astype(float)[..., None, None]

    return np.cos(angle) * np.eye(2) + 1j * np.sin(angle) * PAULI_3

def phase_matrix(phase: (float, np.ndarray)) -> np.ndarray:
    """
    Create a diagonal differential phase matrix or matrices from a (set of) phase(s)

    Inputs;
    - phase: [float, np.ndarray]: the phases to transform into a matrix, shape [...] in case of a np.ndarray

    Outputs:
    - np.ndarray: dtype float, matrix or matrices, shape [..., 2, 2] if phase is a np.ndarray, and [2, 2] if it's a float
    """
    phase = np.array(phase)
    assert phase.dtype in (int, float), f"Phase must have type float, but had type {phase.dtype}"
    phase = phase.astype(float)[..., None, None]

    phasor = np.exp(-0.5j * phase)
    matrix = np.zeros(shape = (*phasor.shape, 2, 2), dtype = complex)
    matrix[..., 0, 0] = phasor
    matrix[..., 1, 1] = phasor.conjugate()
    return matrix