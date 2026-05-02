"""
Constants for use in optical fibre modelling.
"""

from enum import Enum
import sys

import numpy as np
try:
    import cupy as cp
except:
    pass

# Domains
Domain = Enum('Domain', [
    'TIME',
    'FREQUENCY'
])

# Devices
Device = Enum('Device', [
    'CPU',
    'CUDA'
])

# Birefringence modulus models
ModulusModel = Enum('ModulusModel', [
    'RANDOM',
    'FIXED'
])

# Type of gain
Gain = Enum('Gain', [
    'POWER',
    'AMPLITUDE'
])

# Pauli spin matrices
PAULI_1 = np.array([
    [1,  0],
    [0, -1]
], dtype = complex)

# Pauli y spin matrix
PAULI_2 = np.array([
    [0, 1],
    [1, 0]
], dtype = complex)

# Pauli z spin matrix
PAULI_3 = np.array([
    [ 0, -1j],
    [1j,   0]
], dtype = complex)

# Pauli spin matrix vector
PAULI_VECTOR = np.stack([PAULI_1, PAULI_2, PAULI_3])

# Preload variables in GPU memory
if 'cupy' in sys.modules:
    PAULI_1_CUDA      = cp.array(PAULI_1)
    PAULI_2_CUDA      = cp.array(PAULI_2)
    PAULI_3_CUDA      = cp.array(PAULI_3)
    PAULI_VECTOR_CUDA = cp.array(PAULI_VECTOR)