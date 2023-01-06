import logging
import os
from typing import List, Tuple

import h5py
from caiman.motion_correction import MotionCorrect, HAS_CUDA
import caiman as cm
from numpy.typing import NDArray



# EIther split file beforehand, or run moco on whole file, then set x_coords_els and y_coords_els elements to 0
# at frames where no MoCo should be performed. Do not save this first run of moco, but take the shifts, and create
# a new moco object by applying apply_shifts_movie(). Disadvantage: maybe parts of seizure/SD are in a chunk, or
# coming out of an SD, there will be a large jump?

class MotionCorrectSplit(object):
    def __init__(self,
                 fname,
                 min_mov=None,
                 dview=None,
                 max_shifts=(6, 6),
                 niter_rig=1,
                 splits_rig=14,
                 num_splits_to_process_rig=None,
                 strides=(96, 96),
                 overlaps=(32, 32),
                 splits_els=14,
                 num_splits_to_process_els=None,
                 upsample_factor_grid=4,
                 max_deviation_rigid=3,
                 shifts_opencv=True,
                 nonneg_movie=True,
                 gSig_filt=None,
                 use_cuda=False,
                 border_nan=True,
                 pw_rigid=False,
                 num_frames_split=80,
                 var_name_hdf5='mov',
                 is3D=False,
                 indices=(slice(None), slice(None)),
                 split_begin_end_frames: List[Tuple[int, int]] = None,
                 flag_moco: List[bool] = None,
                 temp_path: str = None,
                 np_data_opt: NDArray = None):

        if 'ndarray' in str(type(fname)):
            logging.info('Creating file for motion correction "tmp_mov_mot_corr.hdf5"')
            cm.movie(fname).save('tmp_mov_mot_corr.hdf5')
            fname = ['tmp_mov_mot_corr.hdf5']

        if not isinstance(fname, list):
            fname = [fname]

        if isinstance(gSig_filt, tuple):
            gSig_filt = list(gSig_filt)  # There are some serializers down the line that choke otherwise

        self.fname = fname
        self.dview = dview
        self.max_shifts = max_shifts
        self.niter_rig = niter_rig
        self.splits_rig = splits_rig
        self.num_splits_to_process_rig = num_splits_to_process_rig
        self.strides = strides
        self.overlaps = overlaps
        self.splits_els = splits_els
        self.num_splits_to_process_els = num_splits_to_process_els
        self.upsample_factor_grid = upsample_factor_grid
        self.max_deviation_rigid = max_deviation_rigid
        self.shifts_opencv = bool(shifts_opencv)
        self.min_mov = min_mov
        self.nonneg_movie = nonneg_movie
        self.gSig_filt = gSig_filt
        self.use_cuda = bool(use_cuda)
        self.border_nan = border_nan
        self.pw_rigid = bool(pw_rigid)
        self.var_name_hdf5 = var_name_hdf5
        self.is3D = bool(is3D)
        self.indices = indices
        if self.use_cuda and not HAS_CUDA:
            logging.debug("pycuda is unavailable. Falling back to default FFT.")

        if split_begin_end_frames is not None:  # video should be split in parts, perform moco on split as flag_moco
            # list implies
            # check parameters match
            assert len(flag_moco) == len(split_begin_end_frames), \
                f"Number of parts to split into do not match (split_begin_end_frames: " \
                f"{len(split_begin_end_frames)}, flag_moco: {len(flag_moco)}"
            # TODO: assert moco parts do not share frames? all flag elements bool, split_begin_end_frames int?
        self.split_begin_end_frames = split_begin_end_frames
        self.flag_moco = flag_moco
        if temp_path is not None:  # location to save temp hdf5 files
            self.temp_path = temp_path
        else:
            self.temp_path = \
                os.path.split(fname)[0]  # fixme: assert that fname ends with extension for proper splitting.
            print("temp_path not specified. Saving to folder of video: {self.temp_path}")

        self.np_data_opt = np_data_opt  # optional alternative to already existing hdf5 file. Then fname will be the
        # root file name for the temporary files.

    def split_in_parts(self):
        root_fname = os.path.splitext(self.fname)[0]  # root_fname should contain folder (absolute file path)
        if self.split_begin_end_frames is None or len(self.split_begin_end_frames) == 1:
            fnames = [root_fname + "split.hdf5"]  # TODO: add datetime
        else:
            fnames = [root_fname + str(i).zfill(3) + ".hdf5" for i in range(len(self.split_begin_end_frames))]
        # open nd2 or hdf5 file, check what I get from ripple noise removal. maybe supply numpy array as another
        # class attribute?
        if self.np_data_opt is not None:  # numpy array is data to be saved
            # assert indices match file size
            assert min([min(pair) for pair in self.split_begin_end_frames]) >= 0
            assert max([max(pair) for pair in self.split_begin_end_frames]) < self.np_data_opt.shape[0]  # TODO: check if 0 is the t axis (number of frames)
            # save to hdf5 files
            for i_split, export_fname in enumerate(fnames):
                with h5py.File(export_fname, "w") as hfile:
                    dset =

        # save hdf5 files



        with h5py.File("mytestfile.hdf5", "w") as f:
            dset = f.create_dataset("mydataset", (100,), dtype=)
