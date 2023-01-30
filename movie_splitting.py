import logging
import os

import caiman
import h5py
from caiman.mmapping import prepare_shape
from numpy import float64, ndarray
from typing import List, Optional
import numpy as np


def numpy_to_hdf5(numpy_data: ndarray,
                  export_fpaths,
                  start_end_frames: List[tuple] = None,
                  dataset_name: str = "mov") -> List[str]:
    """
    :param numpy_data: numpy.ndarray. First dimension should be number of frames, e.g. shape should be
    (10000, 512, 512) for an512x512 video of 10000 frames.
    :param export_fpaths: str or list(str). If one export file is to be created, either ["path"] or "path" works.
            If export_fpaths is str and start_end_frames is given, the numbering _1, _2, ... will be applied to the
            export file name. The extension should be
    :param start_end_frames: list of start and end frames of individual parts numpy_data will be split into. A list of
            tuple or list is expected, each element length 2. If elements are longer than 2, the first and second are
            taken as start and end. E.g. if start_end_frames = [(0, 1000)], then there will be one output hdf5 file,
            with the frames 0 to 1000 (including frame 1000, which is the 1001st frame of the video) included.
    :param dataset_name: name of the dataset in the resulting hdf5 file. CaImAn by default uses 'mov', also the default
            in this function. See also 'var_name_hdf5' in various CaImAn functions.
    :return: The list of created files as a list of string (full path with filename and extension)
    """
    FUNC_NAME = "numpy_to_hdf5: "
    if isinstance(export_fpaths, str):  # need to check whether to create multiple file names from single given
        print(f"{FUNC_NAME}Single output filename detected.")
        if isinstance(start_end_frames, list):
            if len(start_end_frames) == 0:
                print(f"{FUNC_NAME}start_end_frames was provided, but it is an empty list. Exporting whole file.")
                start_end_frames = [(0, len(numpy_data) - 1)]
                if export_fpaths.endswith(".hdf5"):
                    export_fpaths = [export_fpaths]
                else:
                    export_fpaths = \
                        [export_fpaths + ".hdf5"]  # TODO: more sophisticated code? (what if wrong extension given?)
            elif len(start_end_frames) == 1:  # do not append _1, _2, ...
                if isinstance(start_end_frames[0], tuple) or isinstance(start_end_frames[0], list):
                    print(f"{FUNC_NAME}saving frames {start_end_frames[0][0]} to {start_end_frames[0][1]}")
                    if export_fpaths.endswith(".hdf5"):
                        export_fpaths = [export_fpaths]
                    else:
                        export_fpaths = \
                            [export_fpaths + ".hdf5"]
            else:  # more than one interval is given
                root_fpath = export_fpaths  # take string export_fpaths as root to append to
                if root_fpath.endswith(".hdf5"):
                    root_fpath = os.path.splitext(root_fpath)[0]
                export_fpaths = [root_fpath + f"_{i}.hdf5" for i in range(len(start_end_frames))]
                print(f"{FUNC_NAME}saving to hdf5 files numbered starting with\n\t{export_fpaths[0]}")
        elif start_end_frames is not None:
            raise Exception(
                f"{FUNC_NAME}start_end_frames - wrong type ({type(start_end_frames)}), expected List[tuple] or List["
                f"List]. Ignoring parameter.")
        else:  # no start_end_frames specified or it is None
            if export_fpaths.endswith(".hdf5"):
                export_fpaths = [export_fpaths]
            else:
                export_fpaths = [export_fpaths + ".hdf5"]
    assert isinstance(export_fpaths, List)
    if start_end_frames is None:
        if len(export_fpaths) > 1:
            raise Exception(f"{FUNC_NAME}several file names given ({len(export_fpaths)}) but no start and end frames.")
        elif len(export_fpaths) == 1:
            print(f"{FUNC_NAME}No splitting will be performed.")
            start_end_frames = [(0, len(numpy_data) - 1)]
        else:  # 0 export fpaths
            raise Exception(f"{FUNC_NAME}length of export_fpaths is 0 (no export files specified).")
    # TODO: check if list of export_fpaths has same length as start_end_frames; if not, need to handle appropriately.
    assert len(export_fpaths) == len(start_end_frames)
    assert start_end_frames[-1][1] <= len(numpy_data) - 1  # avoid index error before creating files happens
    # TODO: not sure which is x and y, but should not matter
    export_folder = os.path.split(export_fpaths[0])[0]
    if not os.path.exists(export_folder):
        print(f"Folder does not exist:\n\t{export_folder}\nCreating folder.")
        os.makedirs(export_folder)
    res_1 = numpy_data.shape[1]  # keep original resolution
    res_2 = numpy_data.shape[2]
    print(f"Creating {len(export_fpaths)} file(s):")
    for i_file, export_fpath in enumerate(export_fpaths):
        print(f"\t{export_fpaths[i_file]}")
        n_frames = start_end_frames[i_file][1] - \
                   start_end_frames[i_file][0] + 1  # include first and last frame specified by start_end_frames[i_file]

        f_shape = (n_frames, res_1, res_2)
        with h5py.File(export_fpath, 'w') as hf:
            dataset = hf.create_dataset(dataset_name, shape=f_shape, dtype=numpy_data.dtype)
            i_frame_counter = 0  # TODO: maybe enumerate(range(...)) is more efficient.
            for i_frame in range(start_end_frames[i_file][0], start_end_frames[i_file][1] + 1):
                dataset[i_frame_counter] = numpy_data[i_frame]
                i_frame_counter += 1
    print("Done.")
    return export_fpaths


# TODO: function taken from motion_correction.py, motion_correction_piecewise(). It should be used to create memmap.
def save_memmap_without_moco(base_name=None,
                             fname=None,
                             order='F',
                             var_name_hdf5='mov',
                             indices=(slice(None), slice(None))):
    if base_name is None:
        base_name = os.path.split(fname)[1][:-4]
    dims, T = caiman.source_extraction.cnmf.utilities.get_file_size(fname, var_name_hdf5=var_name_hdf5)
    z = np.zeros(dims)
    dims = z[indices].shape
    shape_mov = (np.prod(dims), T)
    fname_tot: Optional[str] = caiman.paths.memmap_frames_filename(base_name, dims, T, order)
    if isinstance(fname, tuple):
        fname_tot = os.path.join(os.path.split(fname[0])[0], fname_tot)
    else:
        fname_tot = os.path.join(os.path.split(fname)[0], fname_tot)

    np.memmap(fname_tot, mode='w+', dtype=np.float32,
              shape=prepare_shape(shape_mov), order=order)
    logging.info('Saving no-moco file as {}'.format(fname_tot))
