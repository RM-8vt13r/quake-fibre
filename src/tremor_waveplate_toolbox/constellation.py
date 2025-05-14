"""
A class to represent symbol constellations
"""

from configparser import ConfigParser

import numpy as np

class Constellation:
    """
    A class that represents a set of symbols.
    Each symbol has an in-phase and quadrature component, and a transmission probability.
    The constellation is normalised to average unit symbol energy.
    """
    def __init__(self, coordinates: np.ndarray, probabilities: np.ndarray = None):
        """
        Create a new constellation.
        Coordinates are normalised such that the average transmitted power is 1.
        Probabilities are normalised such that they sum to 1.

        inputs:
        - coordinates [np.ndarray], dtype [complex]: complex coordinates of each symbol
        - probabilities [np.ndarray], dtype [float]: probability of each symbol, leave None for uniform probabilities
        """
        self.coordinates_and_probabilities = (coordinates, probabilities)

    def __call__(self, shape: np.ndarray) -> np.ndarray:
        """
        See draw()
        """
        return self.generate(shape)

    def generate(self, size: np.ndarray) -> np.ndarray:
        """
        Randomly draw symbols from this constellation.

        Inputs:
        - size [np.ndarray]: shape of the array in which to return symbols

        Outputs:
        - [np.ndarray]: array with the desired size, filled with symbols from this constellation
        """
        return np.random.default_rng().choice(
            a       = self.coordinates,
            size    = size,
            p       = self.probabilities,
            shuffle = False
        )

    def copy(self):
        return Constellation(self.coordinates.copy(), self.probabilities.copy())

    @property
    def coordinates(self) -> np.ndarray:
        """
        [np.ndarray] list of complex symbol coordinates in this constellation.
        """
        return self.coordinates_and_probabilities[0]

    @coordinates.setter
    def coordinates(self, value: np.ndarray):
        self.coordinates_and_probabilities = (value, self.probabilities)

    @property
    def probabilities(self) -> np.ndarray:
        """
        [np.ndarray] list of complex symbol coordinates in this constellation
        """
        return self.coordinates_and_probabilities[1]

    @probabilities.setter
    def probabilities(self, value: np.ndarray):
        self.coordinates_and_probabilities = (self.coordinates, value)

    @property
    def coordinates_and_probabilities(self):
        """
        [tuple] tuple with two np.ndarrays of the same length: the first with symbol coordinates, and the second with symbol probabilities.
        """
        return self._coordinates_and_probabilities

    @coordinates_and_probabilities.setter
    def coordinates_and_probabilities(self, value):
        assert isinstance(value, (list, tuple, np.ndarray)), f"coordinates_and_probabilities must be a tuple, but was {type(value)}"
        assert len(value) in (1, 2), f"coordinates_and_probabilities must have length 2 (coordinates and probabilities), but this was {len(value)}"
        if len(value) == 1: value = (*value, None)

        coordinates, probabilities = value

        assert isinstance(coordinates, (np.ndarray, list, tuple)),   f"coordinates must be a np.ndarray, but was {type(coordinates)}"
        coordinates = np.asarray(coordinates.copy())
        assert coordinates.dtype in (complex, float, int), f"coordinates must have dtype complex, but this was {coordinates.dtype}"
        coordinates = coordinates.astype(complex)
        assert len(coordinates.shape) == 1, f"coordinates must have a single dimension, but had {len(coordinates.shape)}"

        if probabilities is None: probabilities = np.ones_like(coordinates, dtype = float) / len(coordinates)
        assert isinstance(probabilities, (np.ndarray, list, tuple)), f"probabilities must be a np.ndarray, but was {type(probabilities)}"
        probabilities = np.asarray(probabilities)
        assert probabilities.dtype in (float, int), f"probabilities must have dtype float, but this was {probabilities.dtype}"
        probabilities = probabilities.astype(float)
        assert len(probabilities.shape) == 1, f"probabilities must have a single dimension, but had {len(probabilities.shape)}"
        assert np.all(probabilities >= 0), f"All probabilities must be nonnegative"
        assert np.sum(probabilities) > 0, f"At least one probability must be nonzero"

        assert coordinates.shape == probabilities.shape, f"coordinates and probabilities must have the same length, but were lengths {coordinates.shape[0]} and {probabilities.shape[0]}"

        # Normalise probabilities to sum to 1
        probabilities /= np.sum(probabilities)

        # Normalise symbols to have unit (weighted) average power
        symbol_energies = np.abs(coordinates) ** 2
        mean_symbol_energy = np.dot(probabilities, symbol_energies)
        coordinates /= np.sqrt(mean_symbol_energy)

        self._coordinates_and_probabilities = (coordinates, probabilities)

BPSK  = Constellation([1, -1])
PSK8  = Constellation([np.cos(phase) + 1j * np.sin(phase) for phase in np.arange(8) * np.pi / 4])
QPSK  = Constellation([a + 1j * b for a in (1, -1) for b in (1, -1)])
QAM4  = QPSK
QAM16 = Constellation([a + 1j * b for a in range(-3, 4, 2) for b in range(-3, 4, 2)])
QAM64 = Constellation([a + 1j * b for a in range(-7, 8, 2) for b in range(-7, 8, 2)])
