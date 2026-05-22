"""
Wrapper around the netCDF4 Dataset, with extra functions to easily make bulk dimensions and variables.
"""
import numpy as np
import netCDF4 as nc

def create_dimensions(dataset: nc.Dataset, dimensions: (tuple, list), sizes: (tuple, list) = None) -> nc.Dataset:
    """
    Create dimensions in a netCDF4 Dataset, if the dimensions don't already exist.
    Does not throw an error if a dimension exists already.
    All dimensions are created as unlimited.

    Inputs:
    - dataset [nc.Dataset]: the Dataset
    - dimensions [tuple, list]: list of dimensions (str) to add.
    - sizes [tuple, list]: list of dimension sizes. If None, all dimensions are unlimited.

    Outputs:
    - [nc.Dataset]: the updated Dataset
    """
    if isinstance(dimensions, str): dimensions = [dimensions]
    if sizes is None: sizes = [None] * len(dimensions)
    if isinstance(sizes, int): sizes = [sizes]
    assert len(sizes) == len(dimensions), f"dimensions and sizes must have the same lengths, but had lengths {len(dimensions)} and {len(sizes)}"

    for dimension, size in zip(dimensions, sizes):
        assert isinstance(dimension, str), f"dimensions must be a list of str, but had an element of type {type(dimension)}"
        assert size is None or isinstance(size, int), f"sizes must be a list of int, but had an element of type {type(size)}"
        if dimension not in dataset.dimensions:
            dataset.createDimension(dimension, size)
        else:
            assert dataset.dimensions[dimension].size == size or (dataset.dimension[dimension].isunlimited() and size is None), f"Tried to overwrite an existing {'un' if dataset.dimensions[dimension].isunlimited() else ''}limited dimension {dimension} of size {dataset.dimensions[dimension].size} with {'un' if size is None else ''}limited dimension{" of size " + str(size) if size is not None else ''}"
    
    return dataset

def create_variables(dataset: nc.Dataset, variables: (tuple, list), types: (tuple, list), dimensions: (tuple, list)) -> nc.Dataset:
    """
    Create variables in a netCDF4 Dataset, if the variables don't already exist.
    Does not throw an error if a variable exists already.

    Inputs:
    - dataset [nc.Dataset]: the Dataset
    - variables [tuple, list]: list of variables (str) to add.
    - types [tuple, list]: list of types (str) corresponding to the variables. See documentaion of netCDF4 for a list of types.
    - dimensions [tuple, list]: list of dimensions. Each variable gets assigned these same dimensions.

    Outputs:
    - [nc.Dataset]: the updated Dataset
    """
    if isinstance(variables, str): variables = [variables]
    if isinstance(types, str): types = [types] * len(variables)
    if isinstance(dimensions, str): dimensions = [dimensions]
    assert len(variables) == len(types), f"variables and types must have the same lengths, but had lengths {len(variables)} and {len(types)}"

    for variable, type_ in zip(variables, types):
        assert isinstance(variable, str), f"variables must be a list of str, but had an element of type {type(variable)}"
        assert isinstance(type_, str), f"types must be a list of str, but had an element of type {type(type_)}"
        if variable not in dataset.variables:
            dataset.createVariable(variable, type_, dimensions)
        else:
            assert dataset.variables[variable].dtype == np.dtype(type_), f"Tried to overwrite existing variable {variable} of type {dataset.variables[variable].dtype} with a new variable of type {np.dtype(type_)} ({type_})"

    return dataset

def create_attributes(dataset: nc.Dataset, keys: (list, tuple), values: (list, tuple), allow_overwrite: bool = False) -> nc.Dataset:
    """
    Create attributes in a netCDF4 Dataset, if the attributes don't already exist.
    Does not throw an error if an attribute exists already.

    Inputs:
    - dataset [nc.Dataset]: the Dataset
    - keys [tuple, list]: list of attribute names (str) to add.
    - values [tuple, list]: list of attribute values to assign.
    - allow_overwrite [bool]: Whether existing attributes should be allowed to be overwritten with new values.

    Output:
    - [nc.Dataset]: the updated Dataset
    """
    if isinstance(keys, str): keys = [keys]
    if '__len__' not in dir(values): values = [values]
    assert len(keys) == len(values), f"keys and values must have the same lengths, but these were {len(keys)} and {len(values)}"

    for key, value in zip(keys, values):
        if key in dataset.ncattrs() and not np.all(value == dataset.getncattr(key)):
            assert allow_overwrite, f"Tried to overwrite existing attribute \"{key}\", but allow_attribute_overwrite is False"
        dataset.setncattr(key, value)

    return dataset

def create_groups(dataset: nc.Dataset, groups: (tuple, list)) -> nc.Dataset:
    """
    Create groups in a netCDF4 Dataset, if the groups don't already exist.
    Does not throw an error if a group exists already.

    Inputs:
    - dataset [nc.Dataset]: the Dataset
    - groups [tuple, list]: list of groups (str) to add.

    Outputs:
    - dataset [nc.Dataset]: the updated Dataset
    """
    if isinstance(groups, str): groups = [groups]
    for group in groups:
        assert isinstance(group, str), f"groups must be a list of str, but had an element of type {type(group)}"
        if group not in dataset.groups:
            dataset.createGroup(group)

    return dataset

def write_variable(dataset: nc.Dataset, variable: str, data: np.ndarray, step_starts: dict = {}) -> nc.Dataset:
    """
    Write data to a variable in a dataset.

    Inputs:
    - dataset [nc.Dataset]: the Dataset
    - variable [str]: The variable to write to.
    - data [np.ndarray]: The data to write. Must have the same number of dimensions as the variable.
    - step_starts [dict]: {dimension [int, str]: step_start [int]} pairs. For each key, treat dimension index 0 in the data variable as index step_start in the netCDF file. This allows e.g. the gradual saving of a large data arrays, by appending multiple smaller arrays.

    Outputs:
    - [nc.Dataset]: the Dataset
    """
    assert len(data.shape) == len(dataset.variables[variable].dimensions), f"Tried to write to variable {variable} with {len(dataset.variables[variable].dimensions)} dimensions, using data with {len(data.shape)} dimensions."

    dimensions = dataset.variables[variable].dimensions
    dimension_slices = []
    for dimension_index, dimension in enumerate(dimensions):
        if dimension in step_starts.keys() and dimension_index not in step_starts.keys():
            step_start = step_starts[dimension]
        elif dimension not in step_starts.keys() and dimension_index in step_starts.keys():
            step_start = step_starts[dimension_index]
        elif dimension in step_starts.keys() and dimension_index in step_starts.keys():
            raise ValueError(f"step_starts defined both keys '{dimension}' and '{dimension_index}', which point to the same dimension of variable {variable}")
        else:
            step_start = None
        dimension_slices.append(slice(step_start, None))

    dataset.variables[variable][*dimension_slices] = data

    return dataset

def read_variable(dataset: nc.Dataset, variable: str, step_starts: dict = {}, step_stops: dict = {}) -> np.ma.MaskedArray:
    """
    Read data from a variable in a dataset.

    Inputs:
    - dataset [nc.Dataset]: the Dataset
    - variable [str]: The variable to read from.
    - step_starts [dict]: {dimension [int, str]: step_start [int]} pairs. For each key, treat dimension index 0 in the data as index step_start in the netCDF file. This allows the partial loading of a large data array.
    - step_stops [dict]: {dimension [int, str]: step_start [int]} pairs. For each key, treat dimension index -1 in the data as index step_stop - 1 in the netCDF file. This allows the partial loading of a large data array.

    Outputs:
    - [np.ndarray]: The read data array
    """
    dimensions = dataset.variables[variable].dimensions
    dimension_slices = []
    for dimension_index, dimension in enumerate(dimensions):
        if dimension in step_starts.keys() and dimension_index not in step_starts.keys():
            step_start = step_starts[dimension]
        elif dimension not in step_starts.keys() and dimension_index in step_starts.keys():
            step_start = step_starts[dimension_index]
        elif dimension in step_starts.keys() and dimension_index in step_starts.keys():
            raise ValueError(f"step_starts defined both keys '{dimension}' and '{dimension_index}', which point to the same dimension of variable {variable}")
        else:
            step_start = None

        if dimension in step_stops.keys() and dimension_index not in step_stops.keys():
            step_stop = step_stops[dimension]
        elif dimension not in step_stops.keys() and dimension_index in step_stops.keys():
            step_stop = step_stops[dimension_index]
        elif dimension in step_stops.keys() and dimension_index in step_stops.keys():
            raise ValueError(f"step_stops defined both keys '{dimension}' and '{dimension_index}', which point to the same dimension of variable {variable}")
        else:
            step_stop = None

        dimension_slices.append(slice(step_start, step_stop))

    return dataset.variables[variable][*dimension_slices]
