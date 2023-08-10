import numpy as np
import math
PERCENTAGE = 0.08  # take lowest 8%
# TODO: the return array has <win_size> first frames cut out (due to algorithm).
def dfof(data: np.array, win_size: int, axis: int = None) -> np.array:
    """
    :param win_size:
    :param data: 1D or 2D array. If 2D, the axis with longer length will be considered as the individual datasets,
    unless the axis parameter is defined.
    :param axis: (optional) 0 or 1. If None, the axis along which the df over f should be calculated is inferred. If
    0, the 0th axis (i.e. data[:, i] is used to get the ith dataset to compute df over f over). If 1, the 1st axis (
    i.e. df over f is performed over data[i, :] for each i)
    :return: a numpy array of the same size as data.
    """
    if len(data.shape) > 1:
        if len(data.shape) == 2:
            if axis is None:
                dset_time_dim = 0 if data.shape[0] >= data.shape[1] else 1  # perform df over f over the longer dataset dimension
                dset_component_dim = 1 if dset_time_dim == 0 else 0
            else:
                dset_time_dim = axis
            # align data such that first index refers to component, second to time
            if dset_component_dim == 1:
                data = data.T
            # perform df over f
            data_dfof = np.zeros(shape=(data.shape[0], data.shape[1]-win_size), dtype=data.dtype)
            for i_component in range(data.shape[dset_component_dim]):  # loop over recordings
                for i_time in range(win_size,data.shape[dset_time_dim]):
                    F0 = np.sort(data[i_component][i_time - win_size:i_time])[math.ceil(PERCENTAGE*win_size)]  # get 8th percentage (lowest) signal
                    data_dfof[i_component][i_time-win_size] = (data[i_component][i_time] - F0)/F0
        else:
            raise Exception(f"dfOverF only implemented for arrays <= 2D")
    else:  # data is one-dimensional
        data_dfof = np.zeros(shape=data.shape[0]-win_size, dtype=data.dtype)
        for i_time in range(win_size, len(data)):
            F0 = np.sort(data[i_time - win_size:i_time])[math.ceil(0.08 * win_size)]  # get 8th percentage (lowest) signal
            data_dfof[i_time-win_size] = (data[i_time] - F0) / F0
    return data_dfof

