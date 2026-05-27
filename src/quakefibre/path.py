"""
A class representing a path over the earth's surface
"""
from typing import override

import numpy as np
import obspy as op
import netCDF4 as nc

from .constants import Dimension
from .dataset import create_dimensions, create_variables, read_variable, write_variable

class Path:
    def __init__(self, longitudes: np.ndarray = None, latitudes: np.ndarray = None, lengths: np.ndarray = None):
        """
        Create the path.

        Inputs:
        - [np.ndarray] longitudes: list of coordinate longitudes in degrees, shape [C,]; if None, lengths must be set
        - [np.ndarray] latitudes: list of coordinate latitudes in degrees, shape [C,]; if None, lengths must be set
        - [np.ndarray] lengths: list of edge lengths in km, shape [C-1,]; if None, longitudes and latitudes must be set
        """
        if lengths is None:
            assert longitudes is not None and latitudes is not None, f"If lengths is None, longitudes and latitudes must be defined"
            assert isinstance(longitudes, (np.ndarray, list, tuple)), f"longitudes must be np.ndarray, but was {type(longitudes)}"
            assert isinstance(latitudes, (np.ndarray, list, tuple)), f"latitudes must be np.ndarray, but was {type(latitudes)}"
            longitudes = np.array(longitudes)
            latitudes = np.array(latitudes)
            assert len(longitudes.shape) == 1, f"longitudes must have shape [C,], but had shape {longitudes.shape}"
            assert latitudes.shape == longitudes.shape, f"longitudes and latitudes must have the same shapes, but had shapes {longitudes.shape} and {latitudes.shape}" 
            self._longitudes = longitudes
            self._latitudes = latitudes

            self._lengths = np.array([
                op.geodetics.base.calc_vincenty_inverse(latitude1, longitude1, latitude2, longitude2, f = 0)[0] / 1000 # f = 0 assumes a perfectly spherical earth. Removing this causes spans to have different lengths.
                for longitude1, latitude1, longitude2, latitude2 in zip(longitudes[:-1], latitudes[:-1], longitudes[1:], latitudes[1:])
            ])

        elif longitudes is None and latitudes is None:
            assert lengths is not None, f"If longitudes and latitudes are None, lengths must be defined"
            assert isinstance(lengths, (np.ndarray, list, tuple)), f"lengths must be np.ndarray, but was {type(lengths)}"
            lengths = np.array(lengths)
            assert len(lengths.shape) == 1, f"lengths must have shape [C,], but had shape {lengths.shape}"
            assert np.all(lengths > 0), f"All lengths must be larger than 0"
            self._lengths = lengths.copy()

            self._longitudes = None
            self._latitudes = None

        else:
            raise ValueError("If lengths is not None, longitudes and latitudes must be None, but they weren't")

    def interpolated(self, positions: (np.ndarray, float)):
        """
        Create a new path from this one by linear spline interpolation.

        Inputs:
        - positions [np.ndarray, float]: if np.ndarray, the positions along the fibre in km where to place vertices on the interpolated path. If float, the distance between vertices on the interpolated path.

        Outputs:
        - [Path]: the interpolated path.
        """
        assert isinstance(positions, (int, np.integer, float, np.floating, np.ndarray, list, tuple)), f"positions must be np.ndarray or float, but was {type(positions)}"

        if isinstance(positions, (int, np.integer, float, np.floating)):
            positions = np.arange(0, self.length, positions)
            positions = np.append(positions, self.length)

        if self._longitudes is None:
            return Path(lengths = np.diff(positions))

        longitudes = np.interp(positions, self.positions, self.longitudes)
        latitudes  = np.interp(positions, self.positions, self.latitudes)
        return Path(longitudes, latitudes)

    def copy(self):
        """
        Copy this path.

        Outputs:
        - [Path] the copied path
        """
        if self.longitudes is None:
            return Path(lengths = self.lengths.copy())

        return Path(
            self.longitudes.copy(),
            self.latitudes.copy()
        )

    def save(self, dataset: nc.Dataset, step_start: int = None) -> nc.Dataset:
        """
        Save this Path in a file as a netCDF4 Dataset

        Inputs:
        - dataset [nc.Dataset]: Path-like or string to the file to save in.
        - step_start [int]: Treat edge index 0 in this Path as index step_start in the netCDF file. This allows e.g. the gradual saving of a large Path, by appending multiple smaller Paths.
        """
        create_dimensions(dataset, Dimension.STEPS.name)
        create_variables(dataset, 'lengths', 'f4', Dimension.STEPS.name)
        write_variable(dataset, 'lengths', self.lengths, {Dimension.STEPS.name: step_start})

        if self.longitudes is not None:
            create_dimensions(dataset, Dimension.VERTICES.name)
            create_variables(dataset, ['longitudes', 'latitudes'], 'f8', Dimension.VERTICES.name)
            write_variable(dataset, 'longitudes', self.longitudes, {Dimension.VERTICES.name: step_start})
            write_variable(dataset, 'latitudes', self.latitudes, {Dimension.VERTICES.name: step_start})

        dataset.sync()

    @classmethod
    def load(cls, dataset: nc.Dataset, step_start: int = None, step_stop: int = None):
        """
        Load a Path from a netCDF4 Dataset

        inputs:
        - dataset [nc.Dataset]: the Dataset to load the Path from
        - step_start [int]: Treat edge index 0 in this Path as index step_start in the netCDF file. This allows the partial loading of a large Path.
        - step_stop [int]: Treat edge index -1 in this Path as index step_stop - 1 in the netCDF file. This allows the partial loading of a large Path.
        
        outputs:
        - [Path]: the loaded Path
        """
        if 'longitudes' in dataset.variables:
            path = Path(
                    longitudes = np.array(read_variable(dataset, 'longitudes', {Dimension.VERTICES.name: step_start}, {Dimension.VERTICES.name: step_stop + 1 if step_stop is not None else None})),
                    latitudes = np.array(read_variable(dataset, 'latitudes', {Dimension.VERTICES.name: step_start}, {Dimension.VERTICES.name: step_stop + 1 if step_stop is not None else None}))
                )

            assert np.allclose(path.lengths, np.array(read_variable(dataset, 'lengths', {Dimension.STEPS.name: step_start}, {Dimension.STEPS.name: step_stop}))), f"lengths of loaded Path do not match lengths saved in the dataset{f" from edge {step_start}" if step_start is not None else ""}{f" to edge {step_stop}" if step_stop is not None else ""}"

        else:
            path = Path(
                    lengths = np.array(read_variable(dataset, 'lengths', {Dimension.STEPS.name: step_start}, {Dimension.STEPS.name: step_stop}))
                )

        return path

    def __iter__(self):
        """
        Prepare iteration over this path's edges.
        """
        self._edge_index = 0
        return self

    def __next__(self):
        """
        Obtain the next edge's subpath

        Output:
        - [Path] the subpath
        """
        if self._edge_index >= self.edge_count:
            raise StopIteration
        self._edge_index += 1
        return self[self._edge_index - 1]

    def __getitem__(self, index):
        """
        Obtain a sub-path

        Input:
        - index [int, slice]: the index or indices of the path edges to include

        Output:
        - [Path] The sliced path
        """
        if self._longitudes is not None:
            if isinstance(index, (int, np.integer)):
                vertex_index = slice(index, index + 2, 1)

            else:
                vertex_index = slice(index.start, index.stop + 1, index.step)

            return Path(
                    self.longitudes[vertex_index],
                    self.latitudes[vertex_index]
                )

        else:
            if isinstance(index, (int, np.integer)):
                edge_index = slice(index, index + 1, 1)

            return Path(
                    lengths = self.lengths[edge_index]
                )

    def __len__(self):
        """
        Obtain the number of Path edges.

        Output:
        - [int] the number of Path edges
        """
        return self.edge_count

    def __eq__(self, other):
        return self.vertex_count == other.vertex_count and \
            ((self._longitudes is None and other._longitudes is None) or np.allclose(self._longitudes, other._longitudes)) and \
            ((self._latitudes is None and other._latitudes is None) or np.allclose(self._latitudes, other._latitudes)) and \
            np.allclose(self.lengths, other.lengths, atol = 1e-5)
            
    @property
    def lengths(self):
        """
        [np.ndarray] length of each path edge in km, shape [C-1,]
        """
        return self._lengths

    @lengths.setter
    def lengths(self, value):
        raise AttributeError("The path lengths cannot be set directly; create a new path instead")
    
    @property
    def length(self):
        """
        [float] length of the whole path
        """
        return np.sum(self.lengths)

    @lengths.setter
    def lengths(self, value):
        raise AttributeError("The path length cannot be set directly; create a new path instead")

    @property
    def vertex_count(self):
        """
        [int] the number of path vertices
        """
        return self.edge_count + 1

    @vertex_count.setter
    def vertex_count(self, value):
        raise AttributeError("The vertex count cannot be set directly; create a new path instead")

    @property
    def edge_count(self):
        """
        [int] the number of path edges
        """
        return len(self.lengths)

    @edge_count.setter
    def edge_count(self, value):
        raise AttributeError("The edge count cannot be set directly; create a new path instead")

    @property
    def longitudes(self):
        """
        [np.ndarray] Path vertex longitudes in chronological order, shape [C,]
        """
        # if self._longitudes is None:
        #     raise AttributeError("Path was initialised without coordinates")
        return self._longitudes

    @longitudes.setter
    def longitudes(self, value):
        raise AttributeError("The longitudes cannot be set after path creation; create a new path instead")

    @property
    def latitudes(self):
        """
        [np.ndarray] Path vertex latitudes in chronological order, shape [C,]
        """
        # if self._latitudes is None:
        #     raise AttributeError("Path was initialised without coordinates")
        return self._latitudes

    @latitudes.setter
    def latitudes(self, value):
        raise AttributeError("The latitudes cannot be set after path creation; create a new path instead")

    @property
    def coordinates(self):
        """
        [np.ndarray] Path vertex coordinates in chronological order, shape [C, 2] where the last dimension contains longitude, latitude
        """
        return np.stack([self.longitudes, self.latitudes], axis = 1)

    @coordinates.setter
    def coordinates(self, value):
        raise AttributeError("The path coordinates cannot be set after path creation; create a new path instead")

    @property
    def positions(self):
        """
        [np.ndarray] distance along the path between the first vertex and each path vertex, shape [C,]
        """
        return np.append([0], np.cumsum(self.lengths))

    @positions.setter
    def positions(self, value):
        raise AttributeError("The path positions cannot be set directly; create a new path instead")
    
    @property
    def centre_longitudes(self):
        """
        [np.ndarray] Path edge centre longitudes in chronological order, shape [C-1,]
        """
        return (self.longitudes[:-1] + self.longitudes[1:]) / 2

    @centre_longitudes.setter
    def centre_longitudes(self, value):
        raise AttributeError("The centre longitudes cannot be set directly; create a new path instead")

    @property
    def centre_latitudes(self):
        """
        [np.ndarray] Path edge centre latitudes in chronological order, shape [C-1,]
        """
        return (self.latitudes[:-1] + self.latitudes[1:]) / 2

    @centre_latitudes.setter
    def centre_latitudes(self, value):
        raise AttributeError("The centre latitudes cannot be set directly; create a new path instead")

    @property
    def centre_coordinates(self):
        """
        [np.ndarray] Path edge centre coordinates in chronological order, shape [C-1, 2] where the last dimension contains longitude, latitude
        """
        return (self.coordinates[:-1] + self.coordinates[1:]) / 2

    @centre_coordinates.setter
    def centre_coordinates(self, value):
        raise AttributeError("The path centre coordinates cannot be set directly; create a new path instead")

    @property
    def centre_positions(self):
        """
        [np.ndarray] distance along the path between the first vertex and each path edge centre, shape [C,]
        """
        positions = self.positions
        return (positions[:-1] + positions[1:]) / 2

    @centre_positions.setter
    def centre_positions(self, value):
        raise AttributeError("The path centre positions cannot be set directly; create a new path instead")
    