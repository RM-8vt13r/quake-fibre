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

from .constants import Domain
from .signal import Signal

class Perturbation(Signal):
    def __init__(self,
                birefringence_scalars: np.ndarray = None,
                birefringence_adders: np.ndarray = None,
                major_angles_adders: np.ndarray = None,
                sample_rate: float = 1,
                domain: Domain = Domain.TIME
            ):
        """
        Create a new Signal.

        Inputs:
        - birefringence_scalars [np.ndarray] or [cp.ndarray]: None, or values to scale birefringence by, shape [K,T] with K fibre sections and T time samples.
        - birefringence_adders [np.ndarray] or [cp.ndarray]: None, or values to add to birefringence after scaling, shape [K,T] with K fibre sections and T time samples.
        - major_angles_adders [np.ndarray] or [cp.ndarray]: None, or values to add to major birefringence axes angles, shape [K,T] with K fibre sections and T time samples.
        - sample_rate [float]: the sample frequency in Hz.
        - domain [Domain]: domain (time or frequency) in which samples is given.
        """
        self._perturbation_presence = []
        samples = None # np.zeros(shape = [1, 1, 3], dtype = float)
        for index, array in enumerate((birefringence_scalars, birefringence_adders, major_angles_adders)):
            if array is None:
                self._perturbation_presence.append(False)
                continue

            if samples is None: samples = np.zeros(shape = [*array.shape, 3], dtype = float)

            try:
                samples[:, :, index] = samples[:, :, index] + array
                self._perturbation_presence.append(True)
            except:
                raise AssertionError(f"birefringence_scalars, birefringence_adders and major_angles_adders must be None or have the same shapes [K, T], but had shapes {[None if a is None else a.shape for a in (birefringence_scalars, birefringence_adders, major_angles_adders)]}")

        assert np.any(self._perturbation_presence), "Cannot create an empty perturbation signal"

        super().__init__(
                samples = samples,
                sample_rate = sample_rate,
                sample_axis = -2,
                domain = domain,
                carrier_wavelength = np.inf
            )

    @property
    def birefringence_scalars(self):
        """
        [np.ndarray, cp.ndarray] The birefringence scalars of this Perturbation in the time domain, shape [K, T], or None if this perturbation is not present
        """
        return self.samples_time[:, :, 0].real if self._perturbation_presence[0] else None

    @birefringence_scalars.setter
    def birefringence_scalars(self, value):
        raise AttributeError("Cannot set birefringence_scalars after creating the Perturbation; make a new Perturbation instead")

    @property
    def birefringence_adders(self):
        """
        [np.ndarray, cp.ndarray] The birefringence adders of this Perturbation in the time domain, shape [K, T], or None if this perturbation is not present
        """
        return self.samples_time[:, :, 1].real if self._perturbation_presence[1] else None

    @birefringence_adders.setter
    def birefringence_adders(self, value):
        raise AttributeError("Cannot set birefringence_adders after creating the Perturbation; make a new Perturbation instead")

    @property
    def major_angles_adders(self):
        """
        [np.ndarray, cp.ndarray] The major angles adders of this Perturbation in the time domain, shape [K, T], or None if this perturbation is not present
        """
        return self.samples_time[:, :, 2].real if self._perturbation_presence[2] else None

    @major_angles_adders.setter
    def major_angles_adders(self, value):
        raise AttributeError("Cannot set major_angles_adders after creating the Perturbation; make a new Perturbation instead")