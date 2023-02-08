import numpy as np
import pims_nd2  # pip install pims_nd2
from typing import Tuple, List

# Comment for testing
# TODO: move this function to a more general library. Possibly: combine with numpy to hdf5 function (
#  movie_splitting.py), in a library called extension_manager, or io (combining with file_handling in this case)

def np_arr_from_nd2(nd2_fpath: str, begin_end_frames: Tuple[int, int]=None):
    # set iter_axes to "t"
    # then: create nd array with sizes matching frame size,
	# begin_end_frames are 1-indexed, i.e. frame 1, 2, ...
    with pims_nd2.ND2_Reader(nd2_fpath) as nikon_file:  # todo: get metadata too?

        sizes_dict = nikon_file.sizes
        if begin_end_frames is not None:
            sizes = (begin_end_frames[1] - begin_end_frames[0] + 1, sizes_dict['x'], sizes_dict['y'])
        else:
            sizes = (sizes_dict['t'], sizes_dict['x'], sizes_dict['y'])
            begin_end_frames = (1, sizes_dict['t'])

        # dtype would be float32 by default...
        frames_arr = np.zeros(sizes, dtype=nikon_file.pixel_type)

        # TODO: probably it is not even necessary to export an np.array, as nikon_file is an iterable of
        #  subclasses of np array... not sure what caiman needs
        i_arrrame = 0
        for i_frame, frame in enumerate(nikon_file):
            # not sure if dtype needed here
            if i_frame >= begin_end_frames[0]-1 and i_frame < begin_end_frames[1]:
                frames_arr[i_arrrame] = np.array(
                    nikon_file[i_frame], dtype=nikon_file.pixel_type)
                i_arrrame += 1
        return frames_arr

def np_arr_and_time_stamps_from_nd2(nd2_fpath: str, begin_end_frames: List[int] = None):
    """
    :param nd2_fpath: str
    :param begin_end_frames: (Optional) in 1-indexing, the indices of the first and last frames of the sequence to be
    imported
    :return:
    frames_arr: 3D array of dims (T, X, Y)
    tstamps_arr: 1x(length of segment) np.float64 array of time stamps in float (ms)
    """
    # set iter_axes to "t"
    # then: create nd array with sizes matching frame size,
    with pims_nd2.ND2_Reader(nd2_fpath) as nikon_file:  # todo: get metadata too?
        # get begin and end frames in 0-indexing
        sizes_dict = nikon_file.sizes
        if begin_end_frames is not None:
            i_begin = begin_end_frames[0] - 1
            i_end = begin_end_frames[1] - 1
        else:
            print(f"Begin and end frames not provided; reading whole video ({sizes_dict['t']})")
            i_begin = 0
            i_end = nikon_file.sizes['t'] - 1
        sizes = (i_end - i_begin + 1, sizes_dict['x'], sizes_dict['y'])

        # dtype would be float32 by default...
        frames_arr = np.zeros(sizes, dtype=nikon_file.pixel_type)
        tstamps_arr = np.zeros(sizes[0], dtype=np.float64)
        # TODO: probably it is not even necessary to export an np.array, as nikon_file is an iterable of
        #  subclasses of np array... not sure what caiman needs
        i_arr_element = 0  # frames_arr should be filled from 0, but the segment might not start from frame 0
        for i_frame in range(i_begin, i_end + 1):
            # not sure if dtype needed here
            frames_arr[i_arr_element] = np.array(
                nikon_file[i_frame], dtype=nikon_file.pixel_type)
            tstamps_arr[i_arr_element] = nikon_file[i_frame].metadata["t_ms"]
            i_arr_element += 1
        return frames_arr, tstamps_arr