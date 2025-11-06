"""
A class representing a path over the earth's surface
"""
import numpy as np
import obspy as op
class Path:
    def __init__(self, longitudes: np.ndarray = None, latitudes: np.ndarray = None, lengths: np.ndarray = None):
        """
        Create the path.

        Inputs:
        - [np.ndarray] longitudes: list of coordinate longitudes in degrees, shape [C,]; if None, lengths must be set
        - [np.ndarray] latitudes: list of coordinate latitudes in degrees, shape [C,]; if None, lengths must be set
        - [np.ndarray] lengths: list of edge lengths, shape [C,]; if lengths is specified (not None), longitudes and latitudes will be ignored, and coordinate-related Path properties will be unavailable
        """
        if lengths is None:
            longitudes = np.array(longitudes)
            latitudes = np.array(latitudes)
            assert len(longitudes.shape) == 1, f"longitudes must have shape [C,], but had shape {longitudes.shape}"
            assert latitudes.shape == longitudes.shape, f"longitudes and latitudes must have the same shapes, but had shapes {longitudes.shape} and {latitudes.shape}" 
            self._longitudes = longitudes
            self._latitudes = latitudes

            self._lengths = np.array([
                op.geodetics.base.calc_vincenty_inverse(latitude1, longitude1, latitude2, longitude2)[0] / 1000
                for longitude1, latitude1, longitude2, latitude2 in zip(longitudes[:-1], latitudes[:-1], longitudes[1:], latitudes[1:])
            ])

        else:
            lengths = np.array(lengths)
            assert len(lengths.shape) == 1, f"lengths must have shape [C,], but had shape {lengths.shape}"
            self._lengths = lengths.copy()

            self._longitudes = None
            self._latitudes = None

    def to_dict(self):
        """
        Represent this path as a dictionary.

        Outputs:
        - [dict] the dictionary representation of this path
        """
        path_dict = {
            'lengths': self.lengths.tolist()
        }

        if self._longitudes is not None:
            path_dict = path_dict | {
                'longitudes': self.longitudes.tolist(),
                'latitudes': self.latitudes.tolist()
            }

        return path_dict

    @classmethod
    def from_dict(cls, path_dict):
        """
        Instantiate a path from a saved dictionary.

        Inputs:
        - path_dict [dict]: a dictionary created using Path.to_dict()

        Outputs:
        - [Path] the loaded Path instance
        """
        if 'longitudes' in path_dict:
            path = cls(np.array(path_dict['longitudes']), np.array(path_dict['latitudes']))
            path._lengths = np.array(path_dict['lengths'])
            return path

        return Path(lengths = np.array(path_dict['lengths']))

    def __iter__(self):
        """
        Prepare iteration over this path's coordinates.
        """
        self._coordinate_index = 0

    def __next__(self):
        """
        Obtain the next vertex's coordinate (longitude, latitude)

        Output:
        - [np.ndarray] the coordinate
        """
        if self._coordinate_index >= self.vertex_count:
            raise StopIteration
        self._coordinate_index += 1
        return self.coordinates[self._coordinate_index - 1]

    def __getitem__(self, index):
        """
        Obtain a sub-path

        Input:
        - index [int, slice]: the index or indices of the path vertices to include

        Output:
        - [Path] The sliced path
        """
        if isinstance(index, int):
            index = slice(index, index + 1, 1)

        return Path(
            self.longitudes[index],
            self.latitudes[index]
        )

    def __len__(self):
        """
        Obtain the number of vertex coordinates.

        Output:
        - [int] the number of vertex coordinates
        """
        return self.vertex_count

    def __eq__(self, other):
        if self.vertex_count == other.vertex_count and \
            np.all(self._longitudes == other._longitudes) and \
            np.all(self._latitudes == other._latitudes) and \
            np.all(self.lengths == other.lengths):
            return True
        return False

    @property
    def lengths(self):
        """
        [np.ndarray] length of each path edge, shape [C-1,]
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
        if self._longitudes is None:
            raise AttributeError("Path was initialised without coordinates")
        return self._longitudes

    @longitudes.setter
    def longitudes(self, value):
        raise AttributeError("The longitudes cannot be set after path creation; create a new path instead")

    @property
    def latitudes(self):
        """
        [np.ndarray] Path vertex latitudes in chronological order, shape [C,]
        """
        if self._latitudes is None:
            raise AttributeError("Path was initialised without coordinates")
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
        [np.ndarray] Path edge centre coordinates in chronological order, shape [C, 2] where the last dimension contains longitude, latitude
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
    