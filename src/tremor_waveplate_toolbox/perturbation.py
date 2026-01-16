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
                material_strains: np.ndarray = None,
                differential_phase_shifts: np.ndarray = None,
                twists: np.ndarray = None,
                sample_rate: float = 1,
                domain: Domain = Domain.TIME
            ):
        """
        Create a new Signal.

        Inputs:
        - material_strains [np.ndarray] or [cp.ndarray]: None, or values to scale birefringence with, shape [K,T] with K fibre sections and T time samples. Birefringence in fibres is scaled by 1 + photoelasticity * material_strains
        - differential_phase_shifts [np.ndarray] or [cp.ndarray]: None, or values to add to birefringence after scaling, shape [K,T] with K fibre sections and T time samples.
        - twists [np.ndarray] or [cp.ndarray]: None, or values to add to major birefringence axes angles, shape [K,T] with K fibre sections and T time samples.
        - sample_rate [float]: the sample frequency in Hz.
        - domain [Domain]: domain (time or frequency) in which samples is given.
        """
        self._perturbation_presence = []
        samples = None # np.zeros(shape = [1, 1, 3], dtype = float)
        for index, array in enumerate((material_strains, differential_phase_shifts, twists)):
            if array is None:
                self._perturbation_presence.append(False)
                continue

            if samples is None: samples = np.zeros(shape = (*array.shape, 3), dtype = complex)

            try:
                samples[:, :, index] = samples[:, :, index] + array.astype(complex)
                self._perturbation_presence.append(True)
            except:
                raise AssertionError(f"material_strains, differential_phase_shifts and twists must be None or have the same shapes [K, T], but had shapes {[None if a is None else a.shape for a in (material_strains, differential_phase_shifts, twists)]}")

        assert np.any(self._perturbation_presence), "Cannot create an empty perturbation signal"

        super().__init__(
                samples = samples,
                sample_rate = sample_rate,
                sample_axis = -2,
                domain = domain,
                carrier_wavelength = np.inf
            )

    @property
    def material_strains(self):
        """
        [np.ndarray, cp.ndarray] The birefringence scalars of this Perturbation in the time domain, shape [K, T], or None if this perturbation is not present.
        Note that this quantity assumes a photoelasticity of one.
        """
        return self.samples_time[:, :, 0].real if self._perturbation_presence[0] else None

    @material_strains.setter
    def material_strains(self, value):
        raise AttributeError("Cannot set material_strains after creating the Perturbation; make a new Perturbation instead")

    @property
    def differential_phase_shifts(self):
        """
        [np.ndarray, cp.ndarray] The birefringence adders of this Perturbation in the time domain, shape [K, T], or None if this perturbation is not present
        """
        return self.samples_time[:, :, 1].real if self._perturbation_presence[1] else None

    @differential_phase_shifts.setter
    def differential_phase_shifts(self, value):
        raise AttributeError("Cannot set differential_phase_shifts after creating the Perturbation; make a new Perturbation instead")

    @property
    def twists(self):
        """
        [np.ndarray, cp.ndarray] The major angles adders of this Perturbation in the time domain, shape [K, T], or None if this perturbation is not present
        """
        return self.samples_time[:, :, 2].real if self._perturbation_presence[2] else None

    @twists.setter
    def twists(self, value):
        raise AttributeError("Cannot set twists after creating the Perturbation; make a new Perturbation instead")