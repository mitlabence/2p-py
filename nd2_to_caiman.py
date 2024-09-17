import numpy as np
import pims_nd2  # pip install pims_nd2
from typing import Tuple, List
import warnings

# Comment for testing
# TODO: move this function to a more general library. Possibly: combine with numpy to hdf5 function (
#  movie_splitting.py), in a library called extension_manager, or io (combining with file_handling in this case)

def np_arr_from_nd2(nd2_fpath: str, begin_end_frames: Tuple[int, int]=None):
    # set iter_axes to "t"
    # then: create nd array with sizes matching frame size,
	# begin_end_frames are 1-indexed, i.e. frame 1, 2, ...
    # begin_end_frames might be tuple (if single segment) or a list of tuples as sorted subsequent, non-overlapping segments.
    # (1, 5) means frame #1 to frame #5 will be in the array.
    # [(1,5), (7,8)] means frames #1 to #5, followed by #7 and #8 wiill be in the array.
    with pims_nd2.ND2_Reader(nd2_fpath) as nikon_file:  # todo: get metadata too?
        sizes_dict = nikon_file.sizes
        if begin_end_frames is not None:
            if isinstance(begin_end_frames, tuple):
                sizes = (begin_end_frames[1] - begin_end_frames[0] + 1, sizes_dict['x'], sizes_dict['y'])
                begin_end_frames = [begin_end_frames]  # convert to list of tuple(s) for looping
            elif isinstance(begin_end_frames, list):
                if len(begin_end_frames) == 0 or not (isinstance(begin_end_frames[0], tuple) or isinstance(begin_end_frames[0], list)):
                    raise ValueError(f"Expected begin_end_frames tuple of int, list of tuple or list of list, got: {type(begin_end_frames)}")
                n_frames = 0
                n_frames_nd2 = sizes_dict["t"]
                # check proper format of list begin_end_frames:
                # 1. should contain tuples or lists or 2 elements
                # 2. the entries should be int
                # 3. each pair should be either equal (segment of length 1) or ascending order. i.e. beginning frame <= end frame
                # 4. each segment should start after the previous, i.e. all segments are non-overlapping and subsequent
                # 5. any segments that start after the length of the passed recording are ignored (removed from the list)
                # 6. the last segment must end on or before the last frame of the nd2 file
                i_segment = 0
                for begin_end_frame in begin_end_frames:
                    if len(begin_end_frame) != 2:  # 1.
                        raise ValueError(f"Expected begin_end_frames to contain lists or tuples of 2 ints, found {len(begin_end_frame)}: {begin_end_frame}")
                    # TODO: add proper working dtpye check that allows int and np.int*, maybe even np.uint*
                    #if not (np.issubdtype(type(begin_end_frame[0]), int) and np.issubdtype(type(begin_end_frame[1]), int)):  # 2.
                    #    raise ValueError(f"Expected begin_end_frames to contain lists or tuples of ints, found types {type(begin_end_frame[0])}, {type(begin_end_frame[1])}")
                    if begin_end_frame[0] > begin_end_frame[1]:  # 3.
                        raise ValueError(f"begin_end_frames: begin frame greater than end frame: {begin_end_frame}")
                    if i_segment > 0:
                        prev_begin_end_frame = begin_end_frames[i_segment-1]
                        if begin_end_frame[0] <= prev_begin_end_frame[1]:  # 4., previous segment already asserted to have begin <= end
                            raise ValueError(f"begin_end_frames: overlapping segments defined: {prev_begin_end_frame}, {begin_end_frame}")
                    if begin_end_frame[0] > n_frames_nd2:  # 5. in 1-indexing, last index would be n_frames_nd2. Last possible segment starts with frame #n_frames_nd2
                        warnings.warn(f"Segment out of recording length: {begin_end_frame} first and last frame (1-indexing), recording has {n_frames_nd2} frames")
                        break
                    if begin_end_frame[1] > n_frames_nd2:  # 6.
                        warnings.warn(f"Segment cut to recording length: {begin_end_frame} cut to {(begin_end_frame[0], n_frames_nd2)}")
                        n_frames += n_frames_nd2 - begin_end_frame[0] + 1
                        begin_end_frames[i_segment] = (begin_end_frame[0], n_frames_nd2)
                        i_segment += 1  # this segment should still be included
                        break  # no future segments can be added, but avoid raising an accidental error in the checks for next segment
                        # (i.e. segment fails 1-4 results in error, even though the segment will not be included)
                    n_frames += begin_end_frame[1] - begin_end_frame[0] + 1
                    i_segment += 1
                begin_end_frames = begin_end_frames[:i_segment]  # 5.
                sizes = (n_frames, sizes_dict["x"], sizes_dict["y"])
        else:
            sizes = (sizes_dict['t'], sizes_dict['x'], sizes_dict['y'])
            begin_end_frames = [(1, sizes_dict['t'])]  # need list of tuple(s) for looping below

        # dtype would be float32 by default...
        frames_arr = np.zeros(sizes, dtype=nikon_file.pixel_type)
        i_arr_frame = 0  # pointer of current frame in frames_arr to be written

        for begin_end_frame in begin_end_frames:
            # for debugging, make sure we are working with list of tuples or lists (begin, end frame indices)
            assert isinstance(begin_end_frame, tuple) or isinstance(begin_end_frame, list)
            for i_frame in range(begin_end_frame[0]-1, begin_end_frame[1]):  # convert to 0-indexing
                frames_arr[i_arr_frame] = np.array(nikon_file[i_frame], dtype=nikon_file.pixel_type)
                i_arr_frame += 1
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