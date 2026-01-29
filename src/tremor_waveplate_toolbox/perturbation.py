"""
A class describing a physical perturbation on an optical fibre, created by a physical event (see perturbation_event.py)
"""
import logging
logger = logging.getLogger()

import numpy as np
try:
    import cupy as cp
except:
    pass
import scipy as sp

from .constants import Domain
from .signal import Signal

class Perturbation(Signal):
    def __init__(self,
                start_time: float = 0,
                strains: np.ndarray = None,
                twists: np.ndarray = None,
                sample_rate: float = 1,
                domain: Domain = Domain.TIME
            ):
        """
        Create a new Signal.

        Inputs:
        - start_time [float]: Absolute time in seconds at which this perturbation starts
        - strains [np.ndarray] or [cp.ndarray]: None, or material strains imposed on each fibre section over time, shape [K, T] with K fibre steps and T time samples. Birefringence in fibres is scaled by 1 + photoelasticity * strains.
        - twists [np.ndarray] or [cp.ndarray]: None, or additive perturbations in radians to the major birefringence axes angles after each fibre step. Shape [K, T].
        - sample_rate [float]: the sample frequency in Hz.
        - domain [Domain]: domain (time or frequency) in which samples is given.
        """
        self._start_time = start_time

        self._perturbation_presence = []
        samples = None

        self._perturbation_presence.append(strains is not None)
        if self._perturbation_presence[0]:
            assert len(strains.shape) == 2, f"strains must have shape [K, T], but had shape {strains.shape}"
            samples = np.zeros(shape = (*strains.shape, 2), dtype = float)
            samples[:, :, 0] = strains

        self._perturbation_presence.append(twists is not None)
        if self._perturbation_presence[1]:
            assert len(twists.shape) == 2, f"twists must have shape [K, T], but had shape {twists.shape}"
            if samples is None:
                samples = np.zeros(shape = (*twists.shape, 2), dtype = float)
            else:
                assert twists.shape[0] == samples.shape[0], f"twists must have shape [K, T] ({samples.shape[:2]}), but had shape {twists.shape}"
            samples[:, :, 1] = twists

        assert np.any(self._perturbation_presence), "Cannot create an empty perturbation signal"

        super().__init__(
                samples = samples,
                sample_rate = sample_rate,
                sample_axis = -2,
                domain = domain,
                carrier_wavelength = np.inf
            )

    @property
    def start_time(self):
        """
        [float] The absolute time in seconds at which this perturbation starts, corresponding to the first time sample of strains and twists.
        """
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        assert isinstance(value, (float, np.floating, int, np.integer)), f"start_time must be an int or float, but was a {type(value)}"
        self._start_time = float(value)

    @property
    def strains(self):
        """
        [np.ndarray, cp.ndarray] The perturbation-induces material strain at each fibre path edge in the time domain, or None if no strains are present. Shape [K, T] where K is the number of fibre steps and T the number of perturbation time steps.
        """
        return self.samples_time[:, :, 0].real if self._perturbation_presence[0] else None

    @strains.setter
    def strains(self, value):
        raise AttributeError("Cannot set strains directly; make a new Perturbation instead")

    @property
    def twists(self):
        """
        [np.ndarray, cp.ndarray] The major angle orientation offsets of this Perturbation in radians in the time domain, shape [K, T] where K is the number of fibre steps and T the number of perturbation time steps, or None if this perturbation is not present.
        """
        return self.samples_time[:, :, 1].real if self._perturbation_presence[1] else None

    @twists.setter
    def twists(self, value):
        raise AttributeError("Cannot set twists after creating the Perturbation; make a new Perturbation instead")