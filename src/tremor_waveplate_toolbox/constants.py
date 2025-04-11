"""
Constants for use in optical fibre modelling.
"""

from enum import Enum

import numpy as np

# Domains
Domain = Enum('Domain', [
    'TIME',
    'FREQUENCY'
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
