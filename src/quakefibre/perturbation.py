"""
A class describing a physical perturbation on an optical fibre, created by a physical event (see perturbation_event.py)
"""
from typing import override
import logging
logger = logging.getLogger()

import numpy as np
import scipy as sp
import netCDF4 as nc

from .constants import Domain, Device, Dimension
from .signal import Signal
from .utilities import rotation_matrix
from .path import Path
from .dataset import create_attributes, create_dimensions, create_variables, write_variable, read_variable

class Perturbation(Signal):
    def __init__(self,
                start_time: float = 0,
                strains: np.ndarray = None,
                twists: np.ndarray = None,
                sample_rate: float = 1,
                domain: Domain = Domain.TIME
            ):
        """
        Create a new Perturbation.

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
            # samples = np.zeros(shape = (*strains.shape, 6), dtype = float)
            samples = np.zeros(shape = (*strains.shape, 2), dtype = float)
            samples[:, :, 0] = strains

        self._perturbation_presence.append(twists is not None)
        if self._perturbation_presence[1]:
            assert len(twists.shape) == 2, f"twists must have shape [K, T], but had shape {twists.shape}"
            if samples is None:
                # samples = np.zeros(shape = (*twists.shape, 6), dtype = float)
                samples = np.zeros(shape = (*twists.shape, 2), dtype = float)
            else:
                assert twists.shape[0] == samples.shape[0], f"twists must have shape [K, T] ({samples.shape[:2]}), but had shape {twists.shape}"
            samples[:, :, 1]  = twists
            # samples[:, :, 2:] = rotation_matrix(twists).reshape((*samples.shape[:2], 4))

        assert np.any(self._perturbation_presence), "Cannot create an empty perturbation signal"

        super().__init__(
                samples = samples,
                sample_rate = sample_rate,
                sample_axis = -2,
                domain = domain,
                carrier_wavelength = np.inf
            )

    def interpolated(self, original_positions: np.ndarray, new_positions: np.ndarray):
        """
        Interpolate the Perturbation in space to fit a new path, using linear spline interpolation.

        Inputs:
        - original_positions [np.ndarray]: positions along the path where the Perturbation was first obtained
        - new_positions [np.ndarray]: positions along the path to which to interpolate this perturbation

        Outputs:
        - [Perturbation] new Perturbation along new_path
        """
        assert isinstance(original_positions, (list, tuple, np.ndarray)), f"original_positions must be a np.ndarray, but was a {type(original_positions)}"
        assert isinstance(new_positions, (list, tuple, np.ndarray)), f"new_positions must be a np.ndarray, but was a {type(new_positions)}"
        original_positions = np.array(original_positions)
        new_positions = np.array(new_positions)
        assert len(original_positions.shape) == 1, f"original_positions must have one dimension, but had {len(original_positions.shape)}"
        assert len(new_positions.shape) == 1, f"new_positions must have one dimension, but had {len(new_positions.shape)}"
        assert len(original_positions) == self.shape[0], f"original_positions length ({len(original_positions)}) must match the first dimension of this Perturbation ({self.shape})"

        self.to_domain(Domain.TIME)

        new_strains = self.xp.zeros(shape = (len(new_positions), self.shape[1]), dtype = self.strains.dtype) if self.strains is not None else None
        new_twists  = self.xp.zeros(shape = (len(new_positions), self.shape[1]), dtype = self.twists.dtype)  if self.twists  is not None else None
        
        for time_index in range(self.shape[1]):
            if new_strains is not None:
                new_strains[:, time_index] = self.xp.interp(new_positions, original_positions, self.strains[:, time_index])
            if new_twists is not None:
                new_twists[:, time_index]  = self.xp.interp(new_positions, original_positions, self.twists[:, time_index])
        
        return Perturbation(
                self.start_time,
                new_strains,
                new_twists,
                self.sample_rate,
                self.domain
            )

    @override
    def copy(self):
        """
        [Perturbation] return a copy of this perturbation
        """
        return Perturbation(
            start_time = self.start_time,
            strains = self.strains.copy() if self.strains is not None else None,
            twists = self.twists.copy() if self.twists is not None else None,
            sample_rate = self.sample_rate,
            domain = self.domain
        )

    @override
    def __eq__(self, other):
        return super().__eq__(other) and self.start_time == other.start_time

    def save(self, dataset: nc.Dataset, step_start: int = None, sample_start: int = None, allow_attribute_overwrite: bool = False, compression: str = 'zlib', compression_level: int = 4) -> nc.Dataset:
        """
        Save this Perturbation in a file as a netCDF4 Dataset

        Inputs:
        - dataset [nc.Dataset]: The Dataset to save in.
        - step_start [int]: If defined, treat step 0 in this Perturbation as step step_start in the netCDF file. This allows e.g. the gradual saving of a large Perturbation, by appending multiple smaller Perturbations.
        - sample_start [int]: If defined, treat time index 0 in this Perturbation as time samples_start in the netCDF file. This allows e.g. the gradual saving of a large Perturbation, by appending multiple smaller Perturbations.
        - allow_attribute_overwrite [bool]: If False, throws an error when you attempt to overwrite an existing dataset attribute with a new value.
        - compression [str]: Compression algorithm to use for saving. If None, save uncompressed data. See https://unidata.github.io/netcdf4-python/#efficient-compression-of-netcdf-variables
        - compression_level [int]: How aggressively to compress. 0 = no compression. 1 = mild compression (fast, but large file). 9 = aggressive compression (slow, but small file)

        Outputs:
        - [nc.Dataset]: The dataset
        """
        original_device = self.device
        self.to_device(Device.CPU)

        create_dimensions(dataset, (Dimension.STEPS.name, Dimension.SAMPLES.name))
        if self.strains is not None:
            create_variables(dataset, 'strains', 'f4', (Dimension.STEPS.name, Dimension.SAMPLES.name), compression, compression_level)
            write_variable(dataset, 'strains', self.strains, {Dimension.STEPS.name: step_start, Dimension.SAMPLES.name: sample_start})
        if self.twists is not None:
            create_variables(dataset, 'twists', 'f4', (Dimension.STEPS.name, Dimension.SAMPLES.name), compression, compression_level)
            write_variable(dataset, 'twists', self.twists, {Dimension.STEPS.name: step_start, Dimension.SAMPLES.name: sample_start})
        create_attributes(dataset, ('start_time', 'sample_rate', 'domain'), (self.start_time, self.sample_rate, self.domain.name), allow_attribute_overwrite)
        dataset.sync()

        self.to_device(original_device)

        return dataset

    @classmethod
    def load(cls, dataset: nc.Dataset, step_start: int = None, step_stop: int = None, sample_start: int = None, sample_stop: int = None):
        """
        Load a Perturbation from a netCDF4 dataset

        inputs:
        - dataset [nc.Dataset]: the Dataset to load the Perturbation from
        - step_start [int]: Treat step index 0 in this Perturbation as index step_start in the netCDF file. This allows the partial loading of a large Perturbation.
        - step_stop [int]: Treat step index -1 in this Perturbation as index step_stop - 1 in the netCDF file. This allows the partial loading of a large Perturbation.
        - sample_start [int]: Treat time index 0 in this Perturbation as time step_start in the netCDF file. This allows the partial loading of a large Perturbation.
        - sample_stop [int]: Treat time index -1 in this Perturbation as time step_stop - 1 in the netCDF file. This allows the partial loading of a large Perturbation.
        
        outputs:
        - [Perturbation]: the loaded Perturbation
        """
        if 'strains' in dataset.variables:
            strains = np.array(read_variable(dataset, 'strains', {Dimension.STEPS.name: step_start, Dimension.SAMPLES.name: sample_start}, {Dimension.STEPS.name: step_stop, Dimension.SAMPLES.name: sample_stop}))
        else:
            strains = None

        if 'twists' in dataset.variables:
            twists = np.array(read_variable(dataset, 'twists', {Dimension.STEPS.name: step_start, Dimension.SAMPLES.name: sample_start}, {Dimension.STEPS.name: step_stop, Dimension.SAMPLES.name: sample_stop}))
        else:
            twists = None

        return Perturbation(
                strains = strains,
                twists = twists,
                **{key: dataset.getncattr(key) for key in dataset.ncattrs()}
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
