import numpy as np
import pims_nd2  # pip install pims_nd2


# Comment for testing
# TODO: move this function to a more general library. Possibly: combine with numpy to hdf5 function (
#  movie_splitting.py), in a library called extension_manager, or io (combining with file_handling in this case)

def np_arr_from_nd2(nd2_fpath: str):
    # set iter_axes to "t"
    # then: create nd array with sizes matching frame size,
    with pims_nd2.ND2_Reader(nd2_fpath) as nikon_file:  # todo: get metadata too?

        sizes_dict = nikon_file.sizes
        sizes = (sizes_dict['t'], sizes_dict['x'], sizes_dict['y'])

        # dtype would be float32 by default...
        frames_arr = np.zeros(sizes, dtype=nikon_file.pixel_type)

        # TODO: probably it is not even necessary to export an np.array, as nikon_file is an iterable of
        #  subclasses of np array... not sure what caiman needs
        for i_frame, frame in enumerate(nikon_file):
            # not sure if dtype needed here
            frames_arr[i_frame] = np.array(
                nikon_file[i_frame], dtype=nikon_file.pixel_type)
        return frames_arr
