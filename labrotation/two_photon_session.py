import os
from tkinter.filedialog import askopenfilename

from bokeh.models import Range1d

import labrotation.belt_processing as belt_processing
import pyabf as abf  # https://pypi.org/project/pyabf/
import pims_nd2
import pandas as pd
import datetime
import pytz  # timezones
import numpy as np
from copy import deepcopy
import json
import labrotation.nikon_ts_reader as ntsr
import h5py
import warnings
from typing import List

# heuristic value, hopefully valid for all recordings made with the digitizer module
LFP_SCALING_FACTOR = 1.0038
# used for saving time stamps. See
# https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
# for further information. %f is 0-padded microseconds to 6 digits.
DATETIME_FORMAT = "%Y.%m.%d-%H:%M:%S.%f_%z"


# TODO: save to hd5 and open from hd5! Everything except the nikon movie could be saved (and the dataframes). Logic
#       is that we would not need the files anymore, we could combine with caiman results. Or, if we want, we can open
#       the nd2 file.
# TODO: in init_and_process, make functions that convert belt_dict and belt_scn_dict from matlab data to numpy
#       class methods that check if variable to convert is matlab array, if not, does nothing.
# TODO: split up methods into more functions that are easily testable (like getting datetime format from string using
#  DATETIME_FORMAT)
#       and write tests. Example: test various datetime inputs for reading out from json (what if no timezone is
#       supplied?)
# TODO: make export_json more verbose! (print save directory and file name, for example)
# TODO: make _match_lfp_nikon_stamps to instead of returning time_offs_lfp_nik change self.time_offs_lfp_nik, like
#       _create_nikon_daq_time. Anything against that?
# TODO: also make sure that each function that infers internal parameters can be run several times and not mess
#       up things, i.e. that parameters read are not changed.
# TODO: class private vs. module private (__ vs _):
#  https://stackoverflow.com/questions/1547145/defining-private-module-functions-in-python
# TODO: make inner functions (now _) class private, and not take any arguments. Just document what attributes it reads
#       and what attributes it changes/sets!
# TODO: add test that attributes of twophotonsession opened from json results file match those of using init or
#       init_and_process
# TODO: implement verbose flag to show/hide print() comments.

class TwoPhotonSession:
    """
    Attributes:
        Basic:
            ND2_PATH: str
            ND2_TIMESTAMPS_PATH: str
            LABVIEW_PATH: str
            LABVIEW_TIMESTAMPS_PATH: str
            LFP_PATH: str
            MATLAB_2P_FOLDER: str

        Implied (assigned using basic attributes)
            nikon_movie
            nikon_meta
            lfp_file: pyabf.abf.ABF, the raw lfp data
            belt_dict
            belt_scn_dict
            belt_df
            belt_scn_df
            nikon_daq_time
            lfp_df
            lfp_df_cut
            time_offs_lfp_nik
            belt_params: dict
            lfp_t_start: datetime.datetime
            nik_t_start: datetime.datetime
            lfp_scaling: float = LFP_SCALING_FACTOR defined in __init__
    """

    # IMPORTANT: change the used lfp_scaling to always be the actual scaling used! Important for exporting (saving)
    # the session data for reproducibility

    # TODO:
    def __init__(self, nd2_path: str = None, nd2_timestamps_path: str = None, labview_path: str = None,
                 labview_timestamps_path: str = None,
                 lfp_path: str = None, matlab_2p_folder: str = None, **kwargs):
        """
        Instantiate a TwoPhotonSession object with only basic parameters defined. This is the default constructor as
        preprocessing might take some time, and it might diverge for various use cases in the future.
        :param nd2_path: complete path of nd2 file
        :param nd2_timestamps_path: complete path of nd2 time stamps (txt) file
        :param labview_path: path of labview txt file (e.g. M278.20221028.133021.txt)
        :param labview_timestamps_path: complete path of labview time stamps (e.g. M278.20221028.133021time.txt)
        :param lfp_path: complete path of lfp (abf) file
        :param matlab_2p_folder: folder of matlab scripts (e.g. C:/matlab-2p/)
        :param kwargs: the inferred attributes of TwoPhotonSession can be directly supplied as keyword arguments. Useful
        e.g. for alternative constructors (like building class from saved json file)
        :return: None
        """
        # set basic attributes, possibly to None.
        self.ND2_PATH = nd2_path
        self.ND2_TIMESTAMPS_PATH = nd2_timestamps_path
        self.LABVIEW_PATH = labview_path
        self.LABVIEW_TIMESTAMPS_PATH = labview_timestamps_path
        self.LFP_PATH = lfp_path
        if matlab_2p_folder is not None:
            self.MATLAB_2P_FOLDER = matlab_2p_folder
        else:
            raise ModuleNotFoundError("matlab-2p path was not found")
        # set inferred attributes to default value
        self.nikon_movie = None
        self.nikon_meta = None
        self.lfp_file = None
        self.belt_dict = None
        self.belt_scn_dict = None
        self.belt_df = None
        self.belt_scn_df = None
        self.nikon_daq_time = None
        self.lfp_df = None
        self.lfp_df_cut = None
        self.time_offs_lfp_nik = None
        self.belt_params = None
        self.lfp_t_start = None
        self.nik_t_start = None
        self.lfp_scaling = None
        self.verbose = True  # printing some extra text by default

        # check for optionally supported keyword arguments:
        self._assign_from_kwargs("nikon_movie", kwargs)
        self._assign_from_kwargs("nikon_meta", kwargs)
        self._assign_from_kwargs("lfp_file", kwargs)
        self._assign_from_kwargs("belt_dict", kwargs)
        self._assign_from_kwargs("belt_scn_dict", kwargs)
        self._assign_from_kwargs("belt_df", kwargs)
        self._assign_from_kwargs("belt_scn_df", kwargs)
        self._assign_from_kwargs("nikon_daq_time", kwargs)
        self._assign_from_kwargs("lfp_df", kwargs)
        self._assign_from_kwargs("lfp_df_cut", kwargs)
        self._assign_from_kwargs("time_offs_lfp_nik", kwargs)
        self._assign_from_kwargs("belt_params", kwargs)
        self._assign_from_kwargs("lfp_t_start", kwargs)
        self._assign_from_kwargs("nik_t_start", kwargs)
        self._assign_from_kwargs("lfp_scaling", kwargs)
        self._assign_from_kwargs("verbose", kwargs)

    def _assign_from_kwargs(self, attribute_name: str, kwargs_dict: dict):
        if attribute_name in kwargs_dict.keys():
            setattr(self, attribute_name, kwargs_dict[attribute_name])
        else:
            setattr(self, attribute_name, None)

    @classmethod
    def init_and_process(cls, nd2_path: str = None, nd2_timestamps_path: str = None, labview_path: str = None,
                         labview_timestamps_path: str = None,
                         lfp_path: str = None, matlab_2p_folder: str = None):
        """
        Instantiate a TwoPhotonSession object and perform the
        :param nd2_path: complete path of nd2 file
        :param nd2_timestamps_path: complete path of nd2 time stamps (txt) file
        :param labview_path: complete path of labview txt file (e.g. M278.20221028.133021.txt)
        :param labview_timestamps_path: complete path of labview time stamps (e.g. M278.20221028.133021time.txt)
        :param lfp_path: complete path of lfp (abf) file
        :param matlab_2p_folder: folder of matlab scripts (e.g. C:/matlab-2p/)
        :return: None
        """
        # infer rest of class attributes automatically.
        instance = cls(nd2_path=nd2_path,
                       nd2_timestamps_path=nd2_timestamps_path,
                       labview_path=labview_path,
                       labview_timestamps_path=labview_timestamps_path,
                       lfp_path=lfp_path,
                       matlab_2p_folder=matlab_2p_folder)
        instance._open_data()
        # convert matlab arrays into numpy arrays
        for k, v in instance.belt_dict.items():
            instance.belt_dict[k] = instance._matlab_array_to_numpy_array(
                instance.belt_dict[k])
        for k, v in instance.belt_scn_dict.items():
            instance.belt_scn_dict[k] = instance._matlab_array_to_numpy_array(
                instance.belt_scn_dict[k])

        instance._nikon_remove_na()
        instance._create_nikon_daq_time()  # defines self.nikon_daq_time
        instance._match_lfp_nikon_stamps()  # creates time_offs_lfp_nik
        instance._create_lfp_df(
            time_offs_lfp_nik=instance.time_offs_lfp_nik, cut_begin=0.0, cut_end=instance.nikon_daq_time.iloc[-1])
        instance._create_belt_df()
        instance._create_belt_scn_df()
        return instance

    @classmethod
    def from_json(cls, fpath: str):
        """
        Re-instantiate a TwoPhotonSession from a saved json file.
        :return: A TwoPhotonSession object
        """
        raise NotImplementedError("from_json() is deprecated and should not be used.")
        with open(fpath, "r") as json_file:
            json_dict = json.load(json_file)
            # get basic attributes
            # TODO: handle missing basic attributes as error
            nd2_path = json_dict["ND2_PATH"]
            nd2_timestamps_path = json_dict["ND2_TIMESTAMPS_PATH"]
            labview_path = json_dict["LABVIEW_PATH"]
            labview_timestamps_path = json_dict["LABVIEW_TIMESTAMPS_PATH"]
            lfp_path = json_dict["LFP_PATH"]
            matlab_2p_folder = json_dict["MATLAB_2P_FOLDER"]
        # optional inferred attributes are in json_dict, few of them need modification
        # lfp_t_start has in format .strftime("%Y.%m.%d-%H:%M:%S")
        json_dict["lfp_t_start"] = datetime.datetime.strptime(json_dict["lfp_t_start"], DATETIME_FORMAT)
        # nik_t_start has format .strftime("%Y.%m.%d-%H:%M:%S")
        json_dict["nik_t_start"] = datetime.datetime.strptime(json_dict["nik_t_start"], DATETIME_FORMAT)
        # time_offs_lfp_nik is float, not str
        json_dict["time_offs_lfp_nik"] = float(json_dict["time_offs_lfp_nik"])
        # lfp_scaling is float, not str
        json_dict["lfp_scaling"] = float(json_dict["lfp_scaling"])
        instance = cls(nd2_path=nd2_path,
                       nd2_timestamps_path=nd2_timestamps_path,
                       labview_path=labview_path,
                       labview_timestamps_path=labview_timestamps_path,
                       lfp_path=lfp_path,
                       matlab_2p_folder=matlab_2p_folder, **json_dict)
        # TODO: at this point, following not matching:
        #           nikon_movie: None
        #           lfp_file: None
        #           belt_dict: None
        #           belt_scn_dict: None
        #           belt_params: None
        #           lfp_t_start: datetime, but probably not UTC-localised or something.
        #               correct: 2021-12-02 10:05:39.204000+00:00, instead: 2021-12-02 10:05:39.204000
        #           nik_t_start: same as lfp_t_start

    @classmethod
    def from_hdf5(cls, fpath: str, try_open_files: bool = True):
        # TODO: handle exceptions (missing data)

        with h5py.File(fpath, "r") as hfile:
            basic_attributes = dict()
            for key, value in hfile["basic"].items():
                basic_attributes[key] = value
            instance = cls(nd2_path=basic_attributes["ND2_PATH"][()],
                           nd2_timestamps_path=basic_attributes["ND2_TIMESTAMPS_PATH"][()],
                           labview_path=basic_attributes["LABVIEW_PATH"][()],
                           labview_timestamps_path=basic_attributes["LABVIEW_TIMESTAMPS_PATH"][()],
                           lfp_path=basic_attributes["LFP_PATH"][()],
                           matlab_2p_folder=basic_attributes["MATLAB_2P_FOLDER"][()])
            # assign dictionary-type attributes
            instance.belt_dict = dict()
            instance.belt_scn_dict = dict()
            instance.belt_params = dict()
            for key, value in hfile["inferred"]["belt_dict"].items():
                instance.belt_dict[key] = value[()]
            for key, value in hfile["inferred"]["belt_scn_dict"].items():
                instance.belt_scn_dict[key] = value[()]
            for key, value in hfile["inferred"]["belt_params"].items():
                instance.belt_params[key] = value[()]
            # create dict for dataframes
            nikon_meta_dict = dict()
            for key, value in hfile["inferred"]["nikon_meta"].items():
                nikon_meta_dict[key] = value[()]
            belt_df_dict = dict()
            for key, value in hfile["inferred"]["belt_df"].items():
                belt_df_dict[key] = value[()]
            belt_scn_df_dict = dict()
            for key, value in hfile["inferred"]["belt_scn_df"].items():
                belt_scn_df_dict[key] = value[()]
            lfp_df_dict = dict()
            for key, value in hfile["inferred"]["lfp_df"].items():
                lfp_df_dict[key] = value[()]
            lfp_df_cut_dict = dict()
            for key, value in hfile["inferred"]["lfp_df_cut"].items():
                lfp_df_cut_dict[key] = value[()]
            instance.nikon_meta = pd.DataFrame.from_dict(nikon_meta_dict)
            instance.belt_df = pd.DataFrame.from_dict(belt_df_dict)
            instance.belt_scn_df = pd.DataFrame.from_dict(belt_scn_df_dict)
            instance.lfp_df = pd.DataFrame.from_dict(lfp_df_dict)
            instance.lfp_df_cut = pd.DataFrame.from_dict(lfp_df_cut_dict)

            instance.nikon_daq_time = pd.Series(hfile["inferred"]["nikon_daq_time"][()])

            instance.time_offs_lfp_nik = hfile["inferred"]["time_offs_lfp_nik"][()]
            instance.lfp_t_start = datetime.datetime.strptime(hfile["inferred"]["lfp_t_start"][()], DATETIME_FORMAT)
            instance.nik_t_start = datetime.datetime.strptime(hfile["inferred"]["nik_t_start"][()], DATETIME_FORMAT)
            instance.lfp_scaling = hfile["inferred"]["lfp_scaling"][()]
        if try_open_files:
            try:
                instance.nikon_movie = pims_nd2.ND2_Reader(instance.ND2_PATH)
            except FileNotFoundError:
                print(f"from_hdf5: nd2 file not found:\n\t{instance.ND2_PATH}. Skipping opening.")
            try:
                instance.lfp_file = abf.ABF(instance.LFP_PATH)
            except FileNotFoundError:
                print(f"from_hdf5: abf file not found:\n\t{instance.LFP_PATH}. Skipping opening.")
        return instance

    def _open_data(self):  # TODO: rename this, as this does not only open data, but also processes some data.
        if self.ND2_PATH is not None:
            self.nikon_movie = pims_nd2.ND2_Reader(self.ND2_PATH)
            # TODO: nikon_movie should be closed properly upon removing this class (or does the garbage collector
            #  take care of it?)
        if self.ND2_TIMESTAMPS_PATH is not None:
            try:
                self.nikon_meta = self.drop_nan_cols(
                    pd.read_csv(
                        self.ND2_TIMESTAMPS_PATH, delimiter="\t", encoding="utf_16_le"))
            except UnicodeDecodeError:
                print(
                    "_open_data(): Timestamp file seems to be unusual. Trying to correct it.")
                output_file_path = os.path.splitext(self.ND2_TIMESTAMPS_PATH)[0] + "_corrected.txt"
                ntsr.standardize_stamp_file(
                    self.ND2_TIMESTAMPS_PATH, output_file_path, export_encoding="utf_16_le")
                self.nikon_meta = self.drop_nan_cols(
                    pd.read_csv(
                        output_file_path, delimiter="\t", encoding="utf_16_le"))
                self.ND2_TIMESTAMPS_PATH = output_file_path
        if hasattr(self, "LABVIEW_PATH") and self.LABVIEW_PATH is not None:
            self.belt_dict, self.belt_scn_dict, self.belt_params = belt_processing.beltProcessPipelineExpProps(
                self.LABVIEW_PATH, self.ND2_TIMESTAMPS_PATH, self.MATLAB_2P_FOLDER)

            for key in self.belt_dict.keys():
                try:
                    self.belt_dict[key] = self._matlab_array_to_numpy_array(self.belt_dict[key])
                except Error:
                    print("Warning: belt_dict could not be mapped from matlab to python datatype!")
            for key in self.belt_scn_dict.keys():
                try:
                    self.belt_scn_dict[key] = self._matlab_array_to_numpy_array(self.belt_scn_dict[key])
                except Error:
                    print("Warning: belt_scn_dict could not be mapped from matlab to python datatype!")
            # convert matlab.double() array to numpy array
            try:
                self.belt_params["belt_length_mm"] = self._matlab_array_to_numpy_array(
                    self.belt_params["belt_length_mm"])
            except AttributeError:
                print(
                    f"No conversion of belt_length_mm happened, as belt_params['belt_length_mm'] is type "
                    f"{type(self.belt_params['belt_length_mm'])}")
        if hasattr(self, "LFP_PATH") and self.LFP_PATH is not None:
            self.lfp_file = abf.ABF(self.LFP_PATH)
            self.lfp_scaling = LFP_SCALING_FACTOR

    def drop_nan_cols(self, dataframe: pd.DataFrame):
        to_drop = []
        for column in dataframe.columns:
            if len(dataframe[column].dropna()) == 0:
                to_drop.append(column)
        return dataframe.drop(to_drop, axis='columns')

    def _matlab_array_to_numpy_array(self, matlab_array):
        if type(matlab_array) is np.ndarray:
            return matlab_array
        else:
            return np.array(matlab_array._data)

    def _nikon_remove_na(self):
        # get the stimulation metadata frames
        event_times = self.nikon_meta[self.nikon_meta["Index"].isna() == True]
        # drop non-imaging frames from metadata
        self.nikon_meta.dropna(subset=["Index"], inplace=True)

    def _lfp_movement_raw(self):
        self.lfp_file.setSweep(sweepNumber=0, channel=1)
        return self.lfp_file.sweepX, self.lfp_file.sweepY

    def _lfp_lfp_raw(self):
        self.lfp_file.setSweep(sweepNumber=0, channel=0)
        return self.lfp_file.sweepX, self.lfp_file.sweepY

    def lfp_movement(self):
        """
        Returns columns of the lfp data internal dataframe as pandas Series: a tuple of (t_series, y_series) for lfp
        channel 0 (lfp) data.
        :return: tuple of two pandas Series
        """
        return self.lfp_df["t_mov_corrected"], self.lfp_df["y_mov"]

    def lfp_lfp(self):
        """
        Returns columns of the lfp data internal dataframe as pandas Series: a tuple of (t_series, y_series) for lfp
        channel 1 (movement) data.
        :return: tuple of two pandas Series
        """
        return self.lfp_df["t_lfp_corrected"], self.lfp_df["y_lfp"]

    def labview_movement(self):
        """
        Returns columns of the labview data internal dataframe as pandas Series: a tuple of (t_series, y_series)
        for labView time and speed data.
        :return: tuple of two pandas Series
        """
        return self.belt_df.time_s, self.belt_df.speed

    def _create_nikon_daq_time(self):
        self.nikon_daq_time = self.nikon_meta["NIDAQ Time [s]"]
        if isinstance(self.nikon_daq_time.iloc[0], float):  # no change needed
            pass
        elif isinstance(self.nikon_daq_time.iloc[0], str):
            # if elements are string, they seem to have comma as decimal separator. Need to replace it by a dot.
            self.nikon_daq_time = self.nikon_daq_time.apply(
                lambda s: float(s.replace(",", ".")))
        else:  # something went really wrong!
            raise ValueError(
                f"nikon_daq_time has unsupported data type: {type(self.nikon_daq_time.iloc[0])}")

    def shift_lfp(self, seconds: float = 0.0, match_type: str = "Nikon") -> None:
        """
        Shifts the LFP signal (+/-) by the amount of seconds: an event at time t to time (t+seconds), i.e. a positive
        seconds means shifting everything to a later time.
        match_type: "Nikon" or "zero". "Nikon": match (cut) to the Nikon frames (NIDAQ time stamps). "zero": match to
        0 s, and the last Nikon frame.
        """
        cut_begin = 0.0
        cut_end = self.nikon_daq_time.iloc[-1]
        if match_type == "Nikon":  # TODO: match_type does not explain its own function
            cut_begin = self.nikon_daq_time.iloc[0]
        self._create_lfp_df(self.time_offs_lfp_nik - seconds, cut_begin, cut_end)
        self.time_offs_lfp_nik = self.time_offs_lfp_nik - seconds

    # TODO: this does not actually matches the two, but gets the offset for matching
    def _match_lfp_nikon_stamps(self) -> None:
        if hasattr(self, "lfp_file") and self.lfp_file is not None:
            # time zone of the recording computer
            tzone_local = pytz.timezone('Europe/Berlin')
            tzone_utc = pytz.utc

            lfp_t_start: datetime.datetime = tzone_local.localize(
                self.lfp_file.abfDateTime)  # supply timezone information
            nik_t_start: datetime.datetime = tzone_utc.localize(
                self.nikon_movie.metadata["time_start_utc"])

            # now both can be converted to utc
            lfp_t_start = lfp_t_start.astimezone(pytz.utc)
            nik_t_start = nik_t_start.astimezone(pytz.utc)

            # save these as instance variables
            self.lfp_t_start = lfp_t_start
            self.nik_t_start = nik_t_start

            # FIXME: correcting for time offset is not enough! The nikon DAQ time also does not start from 0!

            time_offs_lfp_nik = nik_t_start - lfp_t_start
            time_offset_sec = time_offs_lfp_nik.seconds

            time_offs_lfp_nik = time_offset_sec + (time_offs_lfp_nik.microseconds * 1e-6)

            # stop process if too much time detected between starting LFP and Nikon recording.
            if time_offs_lfp_nik > 30.0:
                warnings.warn(
                    f"Warning! more than 30 s difference detected between starting the LFP and the Nikon "
                    f"recording!\nPossible cause: bug in conversion to utc (daylight saving mode, "
                    f"timezone conversion).\nlfp: {lfp_t_start}\nnikon: {nik_t_start}\noffset (s): {time_offs_lfp_nik}")

            print(f"Difference of starting times (s): {time_offs_lfp_nik}")
        else:
            time_offs_lfp_nik = None
        self.time_offs_lfp_nik = time_offs_lfp_nik

    def _belt_dict_to_df(self, belt_dict: dict) -> pd.DataFrame:
        """
        This function takes belt_dict or belt_scn_dict and returns it as a dataframe. Some columns with less entries
        are removed!
        """
        if "runtime" in belt_dict:  # only reliable way I know of to differentiate between belt and belt_scn.
            bd = belt_dict.copy()
            bd.pop("runtime")
            bd.pop("tsscn")
            df = pd.DataFrame(bd)
        else:
            df = pd.DataFrame(belt_dict)
        if "time" in df.columns:
            df["time_s"] = df["time"] / 1000.
        return df

    def _create_belt_df(self):
        self.belt_df = self._belt_dict_to_df(self.belt_dict)

    def _create_belt_scn_df(self):
        self.belt_scn_df = self._belt_dict_to_df(self.belt_scn_dict)

    def _create_lfp_cut_df(self, lfp_df_raw, lower_limit: float, upper_limit: float) -> pd.DataFrame:
        lfp_df_new_cut = lfp_df_raw[lfp_df_raw["t_lfp_corrected"]
                                    >= lower_limit]
        lfp_df_new_cut = lfp_df_new_cut[lfp_df_new_cut["t_lfp_corrected"]
                                        <= upper_limit]
        return lfp_df_new_cut

    def _create_lfp_df(self, time_offs_lfp_nik: float, cut_begin: float, cut_end: float):
        if hasattr(self, "lfp_file") and self.lfp_file is not None:
            lfp_df_new = pd.DataFrame()
            lfp_df_new_cut = pd.DataFrame()
            t_mov, y_mov = self._lfp_movement_raw()
            t_lfp, y_lfp = self._lfp_lfp_raw()
            # add movement data
            lfp_df_new["t_mov_raw"] = t_mov
            lfp_df_new["t_mov_offset"] = lfp_df_new["t_mov_raw"] - time_offs_lfp_nik
            # scale factor given in Bence's excel sheets
            lfp_df_new["t_mov_corrected"] = lfp_df_new["t_mov_offset"] * \
                                            LFP_SCALING_FACTOR
            lfp_df_new["y_mov"] = y_mov
            # add normalized movement data
            motion_min = lfp_df_new["y_mov"].min()
            motion_max = lfp_df_new["y_mov"].max()
            motion_mean = lfp_df_new["y_mov"].mean()
            # FIXME: normalization should be (y - mean)/(max - min)
            lfp_df_new["y_mov_normalized"] = lfp_df_new["y_mov"] / motion_mean

            # add lfp data
            lfp_df_new["t_lfp_raw"] = t_lfp
            lfp_df_new["t_lfp_offset"] = lfp_df_new["t_lfp_raw"] - time_offs_lfp_nik
            # scale factor given in Bence's excel sheets
            # TODO: document columns of dataframes. corrected vs offset
            lfp_df_new["t_lfp_corrected"] = lfp_df_new["t_lfp_offset"] * LFP_SCALING_FACTOR
            lfp_df_new["y_lfp"] = y_lfp

            # cut lfp data
            # cut to Nikon. LFP will not start at 0!
            # lfp_df_new_cut = self._create_lfp_cut_df(
            #    lfp_df_new, self.nikon_daq_time.iloc[0], self.nikon_daq_time.iloc[-1])
            lfp_df_new_cut = self._create_lfp_cut_df(
                lfp_df_new, cut_begin, cut_end)
            self.lfp_df, self.lfp_df_cut = lfp_df_new, lfp_df_new_cut
        else:
            print("TwoPhotonSession: LFP file was not specified.")
            self.lfp_df, self.lfp_df_cut = None, None

        # now nd2_to_caiman.py

    def get_nikon_data(self, i_begin: int = None, i_end: int = None) -> np.array:  # TODO: test this
        # set iter_axes to "t"
        # then: create nd array with sizes matching frame size,
        sizes_dict = self.nikon_movie.sizes
        pixel_type = self.nikon_movie.pixel_type
        if (i_begin is not None) and (i_end is not None):
            n_frames = i_end - i_begin
            i_first = i_begin
        else:
            n_frames = sizes_dict['t']
            i_first = 0
        sizes = (n_frames, sizes_dict['x'], sizes_dict['y'])
        # dtype would be float32 by default...
        frames_arr = np.zeros(sizes, dtype=pixel_type)
        for i_frame in range(n_frames):
            frames_arr[i_frame] = np.array(self.nikon_movie[i_first + i_frame],
                                           dtype=pixel_type)  # not sure if dtype needed here
        return frames_arr

    def export_json(self, **kwargs) -> None:
        """
        *args:
            fpath - absolute file path of json file. If not supported, it will be
            nd2 file path and name, with json extension.

        Given an absolute file path fpath to a not existing json file, the important
        parameters of the session are exported into this json file.
        Parameters to serialize:
            Basic attributes:
                self.ND2_PATH
                self.ND2_TIMESTAMPS_PATH
                self.LABVIEW_PATH
                self.LABVIEW_TIMESTAMPS_PATH
                self.LFP_PATH
                self.MATLAB_2P_FOLDER
            Inferred attributes:
                self.belt_dict
                self.belt_scn_dict
                self.time_offs_lfp_nik
                self.lfp_t_start
                self.nik_t_start
                self.lfp_scaling
                belt_params

            Not saved:
                (lfp_file)
                (nikon_movie)
                (nikon_meta)
                (lfp_df)
                (lfp_df_cut)
                (belt_df)
                (belt_scn_df)
                (nikon_daq_time)
        """
        raise NotImplementedError("export_json() is deprecated and should not be used.")
        fpath = kwargs.get("fpath", os.path.splitext(self.ND2_PATH)[0] + ".json")

        if self.belt_params is None:
            params = {}
        else:
            params = deepcopy(self.belt_params)

        # set basic attributes
        params["ND2_PATH"] = self.ND2_PATH
        params["ND2_TIMESTAMPS_PATH"] = self.ND2_TIMESTAMPS_PATH
        params["LABVIEW_PATH"] = self.LABVIEW_PATH
        params["LABVIEW_TIMESTAMPS_PATH"] = self.LABVIEW_TIMESTAMPS_PATH
        params["LFP_PATH"] = self.LFP_PATH
        params["MATLAB_2P_FOLDER"] = self.MATLAB_2P_FOLDER

        params["time_offs_lfp_nik"] = self.time_offs_lfp_nik
        params["lfp_t_start"] = self.lfp_t_start.strftime(DATETIME_FORMAT)
        params["nik_t_start"] = self.nik_t_start.strftime(DATETIME_FORMAT)
        params["lfp_scaling"] = self.lfp_scaling

        # turn everything into a string for json dump
        for k, v in params.items():
            params[k] = str(v)

        with open(fpath, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Saved TwoPhotonSession instance to json file:\n\t{fpath}")

    # TODO: handle missing files
    # FIXME: if lfp missing, no inferred group is created!
    def export_hdf5(self, **kwargs) -> None:
        # set export file name and path
        fpath = kwargs.get("fpath", os.path.splitext(self.ND2_PATH)[0] + ".h5")
        with h5py.File(fpath, "w") as hfile:
            basic_group = hfile.create_group("basic")
            # basic parameters
            basic_group["ND2_PATH"] = self.ND2_PATH
            basic_group["ND2_TIMESTAMPS_PATH"] = self.ND2_TIMESTAMPS_PATH
            basic_group["LABVIEW_PATH"] = self.LABVIEW_PATH
            basic_group["LABVIEW_TIMESTAMPS_PATH"] = self.LABVIEW_TIMESTAMPS_PATH
            basic_group["LFP_PATH"] = self.LFP_PATH if self.LFP_PATH is not None else ""
            basic_group["MATLAB_2P_FOLDER"] = self.MATLAB_2P_FOLDER
            # implied parameters
            inferred_group = hfile.create_group("inferred")
            # save nikon_meta as group with columns as datasets
            nikon_meta_group = inferred_group.create_group("nikon_meta")
            for colname in self.nikon_meta.keys():
                nikon_meta_group[colname] = self.nikon_meta[colname].to_numpy()
            # save belt_dict
            belt_dict_group = inferred_group.create_group("belt_dict")
            for key, value in self.belt_dict.items():
                belt_dict_group[key] = value
            # save belt_scn_dict
            belt_scn_dict_group = inferred_group.create_group("belt_scn_dict")
            for key, value in self.belt_scn_dict.items():
                belt_scn_dict_group[key] = value
            # save pandas Series nikon_daq_time
            inferred_group["nikon_daq_time"] = self.nikon_daq_time.to_numpy()
            # save time_offs_lfp_nik
            inferred_group[
                "time_offs_lfp_nik"] = self.time_offs_lfp_nik if self.time_offs_lfp_nik is not None else np.nan
            # save belt_params
            belt_params_group = inferred_group.create_group("belt_params")
            for key, value in self.belt_params.items():
                belt_params_group[key] = value
            # save lfp_t_start, nik_t_start, lfp_scaling
            inferred_group["lfp_t_start"] = self.lfp_t_start.strftime(
                DATETIME_FORMAT) if self.lfp_t_start is not None else ""
            inferred_group["nik_t_start"] = self.nik_t_start.strftime(
                DATETIME_FORMAT) if self.nik_t_start is not None else ""
            inferred_group["lfp_scaling"] = self.lfp_scaling if self.lfp_scaling is not None else np.nan
            # save lfp_df
            lfp_df_group = inferred_group.create_group("lfp_df")
            if self.lfp_df is not None:
                for colname in self.lfp_df.keys():
                    lfp_df_group[colname] = self.lfp_df[colname].to_numpy()
            # save lfp_df_cut
            lfp_df_cut_group = inferred_group.create_group("lfp_df_cut")
            if self.lfp_df_cut is not None:
                for colname in self.lfp_df_cut.keys():
                    lfp_df_cut_group[colname] = self.lfp_df_cut[colname].to_numpy()
            # save belt_df
            belt_df_group = inferred_group.create_group("belt_df")
            if self.belt_df is not None:
                for colname in self.belt_df.keys():
                    belt_df_group[colname] = self.belt_df[colname].to_numpy()
            # save belt_scn_df
            belt_scn_df_group = inferred_group.create_group("belt_scn_df")
            if self.belt_scn_df is not None:
                for colname in self.belt_scn_df.keys():
                    belt_scn_df_group[colname] = self.belt_scn_df[colname].to_numpy()

    # TODO: get nikon frame matching time stamps (NIDAQ time)! It is session.nikon_daq_time
    def return_nikon_mean(self):
        return [self.nikon_movie[i_frame].mean() for i_frame in range(self.nikon_movie.sizes["t"])]

    def infer_labview_timestamps(self):
        """
        Try to infer the labview time stamps filename given the labview filename.
        :return: None
        """
        if self.LABVIEW_PATH is not None:
            inferred_fpath = os.path.splitext(self.LABVIEW_PATH)[0] + "time.txt"
            if os.path.exists(inferred_fpath):
                if self.LABVIEW_TIMESTAMPS_PATH is None:
                    self.LABVIEW_TIMESTAMPS_PATH = inferred_fpath
                    print(f"Inferred labview timestamps file path:\n\t{self.LABVIEW_TIMESTAMPS_PATH}")
                else:  # timestamps file already defined
                    print(
                        f"Labview timestamps file seems to already exist:\n\t{self.LABVIEW_TIMESTAMPS_PATH}\nNOT "
                        f"changing it.")
        else:
            print(
                "Can not infer labview timestamps filename, as no labview data was defined. (txt file with labview "
                "readout data)")


def open_session(data_path: str) -> TwoPhotonSession:
    # FIXME: askopenfilename from TKinter not working. Put into file_handling?
    # TODO: test this function! Make it an alternative constructor
    # .nd2 file
    nd2_path = askopenfilename(initialdir=data_path, title="Select .nd2 file")
    print(f"Selected imaging file: {nd2_path}")

    # nd2 info file (..._nik.txt) Image Proterties -> Recorded Data of .nd2 file saved as .txt
    nd2_timestamps_path = os.path.splitext(nd2_path)[0] + "_nik.txt"
    if not os.path.exists(nd2_timestamps_path):
        nd2_timestamps_path = askopenfilename(initialdir=data_path,
                                              title="Nikon info file not found. Please provide it!")
    print(f"Selected nd2 info file: {nd2_timestamps_path}")

    # labview .txt file
    labview_path = askopenfilename(
        initialdir=data_path, title="Select corresponding labview (xy.txt) file")
    print(f"Selected LabView data file: {labview_path}")

    # labview time stamp (...time.txt)
    labview_timestamps_path = \
        os.path.splitext(labview_path)[0] + "time.txt"  # try to open the standard corresponding time stamp file first
    if not os.path.exists(labview_timestamps_path):
        labview_timestamps_path = askopenfilename(initialdir=data_path,
                                                  title="Labview time stamp not found. Please provide it!")
    print(f"Selected LabView time stamp file: {labview_timestamps_path}")

    # lfp file (.abf)
    lfp_path = askopenfilename(
        initialdir=data_path, title="Select LFP .abf file")
    print(f"Selected LFP file: {lfp_path}")

    session = TwoPhotonSession(nd2_path=nd2_path, nd2_timestamps_path=nd2_timestamps_path, labview_path=labview_path,
                               labview_timestamps_path=labview_timestamps_path, lfp_path=lfp_path)
    return session


# TODO: extract these methods to a new python file, and move imports outside functions to speed up.
# taken from caiman.utils.visualization.py
def nb_view_patches_with_lfp_movement(Yr, A, C, b, f, d1, d2,
                                      t_lfp: np.array = None, y_lfp: np.array = None,
                                      t_mov: np.array = None, y_mov: np.array = None,
                                      YrA=None, image_neurons=None, thr=0.99, denoised_color=None, cmap='jet',
                                      r_values=None, SNR=None, cnn_preds=None):
    """
    Interactive plotting utility for ipython notebook

    Args:
        Yr: np.ndarray
            movie

        A,C,b,f: np.ndarrays
            outputs of matrix factorization algorithm

        d1,d2: floats
            dimensions of movie (x and y)

        YrA:   np.ndarray
            ROI filtered residual as it is given from update_temporal_components
            If not given, then it is computed (K x T)

        image_neurons: np.ndarray
            image to be overlaid to neurons (for instance the average)

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param r_values:
    """
    from past.utils import old_div
    import matplotlib as mpl
    from scipy.sparse import spdiags
    from caiman.utils.visualization import get_contours
    try:
        import bokeh
        import bokeh.plotting as bpl
        from bokeh.models import CustomJS, ColumnDataSource, Range1d, LabelSet, Slider
        from bokeh.layouts import layout, row, column
    except:
        print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")

    colormap = mpl.cm.get_cmap(cmap)
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    nA2 = np.ravel(np.power(A, 2).sum(0)) if isinstance(A, np.ndarray) else np.ravel(A.power(2).sum(0))
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
                       (A.T * np.matrix(Yr) -
                        (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                        A.T.dot(A) * np.matrix(C)) + C)
    else:
        Y_r = C + YrA

    x = np.arange(T)
    if image_neurons is None:
        image_neurons = A.mean(1).reshape((d1, d2), order='F')

    coors = get_contours(A, (d1, d2), thr)
    cc1 = [cor['coordinates'][:, 0] for cor in coors]
    cc2 = [cor['coordinates'][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    source = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))
    source_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

    code = """
            var data = source.data
            var data_ = source_.data
            var f = cb_obj.value - 1
            var x = data['x']
            var y = data['y']
            var y2 = data['y2']

            for (var i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+f*x.length]
                y2[i] = data_['z2'][i+f*x.length]
            }

            var data2_ = source2_.data;
            var data2 = source2.data;
            var c1 = data2['c1'];
            var c2 = data2['c2'];
            var cc1 = data2_['cc1'];
            var cc2 = data2_['cc2'];

            for (var i = 0; i < c1.length; i++) {
                   c1[i] = cc1[f][i]
                   c2[i] = cc2[f][i]
            }
            source2.change.emit();
            source.change.emit();
        """

    if r_values is not None:
        code += """
            var mets = metrics.data['mets']
            mets[1] = metrics_.data['R'][f].toFixed(3)
            mets[2] = metrics_.data['SNR'][f].toFixed(3)
            metrics.change.emit();
        """
        metrics = ColumnDataSource(data=dict(y=(3, 2, 1, 0),
                                             mets=('', "% 7.3f" % r_values[0], "% 7.3f" % SNR[0],
                                                   "N/A" if np.sum(cnn_preds) in (0, None) else "% 7.3f" % cnn_preds[
                                                       0]),
                                             keys=("Evaluation Metrics", "Spatial corr:", "SNR:", "CNN:")))
        if np.sum(cnn_preds) in (0, None):
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR))
        else:
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR, CNN=cnn_preds))
            code += """
                mets[3] = metrics_.data['CNN'][f].toFixed(3)
            """
        labels = LabelSet(x=0, y='y', text='keys', source=metrics, render_mode='canvas')
        labels2 = LabelSet(x=10, y='y', text='mets', source=metrics, render_mode='canvas', text_align="right")
        plot2 = bpl.figure(plot_width=200, plot_height=100, toolbar_location=None)
        plot2.axis.visible = False
        plot2.grid.visible = False
        plot2.tools.visible = False
        plot2.line([0, 10], [0, 4], line_alpha=0)
        plot2.add_layout(labels)
        plot2.add_layout(labels2)
    else:
        metrics, metrics_ = None, None

    callback = CustomJS(args=dict(source=source, source_=source_, source2=source2,
                                  source2_=source2_, metrics=metrics, metrics_=metrics_), code=code)

    plot = bpl.figure(plot_width=600, plot_height=200, x_range=Range1d(0, Y_r.shape[0]))
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1,
                  line_alpha=0.6, color=denoised_color)

    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr,
                       plot_width=int(min(1, d2 / d1) * 300),
                       plot_height=int(min(1, d1 / d2) * 300))

    plot1.image(image=[image_neurons[::-1, :]], x=0,
                y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1', 'c2', alpha=0.6, color='purple',
                line_width=2, source=source2)

    # create plot for lfp
    if y_lfp is not None:
        source_lfp = ColumnDataSource(data=dict(x=t_lfp, y=y_lfp))
        plot_lfp = bpl.figure(x_range=Range1d(t_lfp[0], t_lfp[-1]),
                              y_range=Range1d(y_lfp.min(), y_lfp.max()),
                              plot_width=plot.plot_width,
                              plot_height=plot.plot_height)
        plot_lfp.line("x", "y", source=source_lfp)
    # plot_mov = bpl.figure(x_range=xr, y_range=None)
    if y_mov is not None:
        source_mov = ColumnDataSource(data=dict(x=t_mov, y=y_mov))
        plot_mov = bpl.Figure(x_range=Range1d(t_mov[0], t_mov[-1]),
                              y_range=Range1d(y_mov.min(), y_mov.max()),
                              plot_width=plot.plot_width,
                              plot_height=plot.plot_height)
        plot_mov.line("x", "y", source=source_mov)
    if Y_r.shape[0] > 1:
        slider = Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                        title="Neuron Number")
        slider.js_on_change('value', callback)
        if y_mov is not None:
            if y_lfp is not None:  # both lfp and mov
                bpl.show(layout([[slider], [row(
                    plot1 if r_values is None else column(plot1, plot2),
                    column(plot, plot_lfp, plot_mov))]]))
            else:  # no lfp plot
                bpl.show(layout([[slider], [row(
                    plot1 if r_values is None else column(plot1, plot2),
                    column(plot, plot_mov))]]))
        else:  # no mov plot
            if y_lfp is not None:
                bpl.show(layout([[slider], [row(
                    plot1 if r_values is None else column(plot1, plot2),
                    column(plot, plot_lfp))]]))
            else:  # no lfp and no movement
                bpl.show(layout([[slider], [row(
                    plot1 if r_values is None else column(plot1, plot2), plot)]]))
    else:
        bpl.show(row(plot1 if r_values is None else
                     column(plot1, plot2), plot))

    return Y_r


# TODO: extract these methods to a new python file, and move imports outside functions to speed up.
# taken from caiman.utils.visualization.py
def nb_view_patches_manual_control(Yr, A, C, b, f, d1, d2,
                                   YrA=None, image_neurons=None, thr=0.99, denoised_color=None, cmap='jet',
                                   r_values=None, SNR=None, cnn_preds=None, mode: str = None, idx_accepted: List = None,
                                   idx_rejected: List = None):
    """
    Interactive plotting utility for ipython notebook

    Args:
        Yr: np.ndarray
            movie

        A,C,b,f: np.ndarrays
            outputs of matrix factorization algorithm

        d1,d2: floats
            dimensions of movie (x and y)

        YrA:   np.ndarray
            ROI filtered residual as it is given from update_temporal_components
            If not given, then it is computed (K x T)

        image_neurons: np.ndarray
            image to be overlaid to neurons (for instance the average)

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param r_values:

        mode: string
            The initial view mode to show. It should be one of 'accepted', 'rejected', 'all'. The fourth option,
            'modified', would cause an empty plot.

        idx_accepted: List
            The idx_components field of the estimates object.

        idx_rejected: List
            The idx_components_bad field of the estimates object.
    """
    from past.utils import old_div
    import matplotlib as mpl
    from scipy.sparse import spdiags
    from caiman.utils.visualization import get_contours
    from labrotation.file_handling import get_filename_with_date
    try:
        import bokeh
        import bokeh.plotting as bpl
        from bokeh.models import CustomJS, ColumnDataSource, Range1d, LabelSet, Dropdown, Slider
        from bokeh.models.widgets.buttons import Button, Toggle
        from bokeh.layouts import layout, row, column
    except:
        print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")
    # TODO: idx_components and idx_components_bad refer to indices of accepted/rejected neurons, use these in
    #  nb_view_components_manual_control. If These don't exist, that means select_components has been called... I don't
    #  know if it is still possible (easily) to move the neurons from one group to the other.

    """
        nb_view_patches_manual_control(
        Yr, estimates.A.tocsc()[:, idx], estimates.C[idx], estimates.b, estimates.f,
        estimates.dims[0], estimates.dims[1],
        YrA=estimates.R[idx], image_neurons=img,
        thr=thr, denoised_color=denoised_color, cmap=cmap,
        r_values=None if estimates.r_values is None else estimates.r_values[idx],
        SNR=None if estimates.SNR_comp is None else estimates.SNR_comp[idx],
        cnn_preds=None if np.sum(estimates.cnn_preds) in (0, None) else estimates.cnn_preds[idx],
        mode=mode)
    """

    # No easy way to use these in CustomJS. Could define beginning of variable 'code' like this, and append the rest
    # REJECTED_COLOR = "red"
    # REJECTED_TEXT = "rejected"
    # ACCEPTED_COLOR = "green"
    # ACCEPTED_TEXT = "accepted"

    # idx_accepted and idx_rejected should be disjoint lists coming from CaImAn. (0-indexing)
    # set to 1 all the entries that correspond to accepted components. Rest is 0.
    cell_category_original = [0 for i in range(len(idx_accepted) + len(idx_rejected))]
    for i_accepted in idx_accepted:
        cell_category_original[i_accepted] = 1
    cell_category_new = cell_category_original.copy()

    colormap = mpl.cm.get_cmap(cmap)
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape

    nA2 = np.ravel(np.power(A, 2).sum(0)) if isinstance(A, np.ndarray) else np.ravel(A.power(2).sum(0))
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        # Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
        #               (A.T * np.matrix(Yr) -
        #                (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
        #                A.T.dot(A) * np.matrix(C)) + C)
        raise NotImplementedError("YrA is None; this has not been implemented yet")
    else:
        Y_r = C + YrA

    x = np.arange(T)
    if image_neurons is None:
        raise NotImplementedError("image_neurons is None; this has not been implemented yet")
        # image_neurons = A.mean(1).reshape((d1, d2), order='F')

    coors = get_contours(A, (d1, d2), thr)
    cc1 = [cor['coordinates'][:, 0] for cor in coors]
    cc2 = [cor['coordinates'][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    source = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))  # contains traces of single neuron
    source_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))  # contains all traces; use this to update source
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))
    categories = ColumnDataSource(data=dict(cats=cell_category_original))
    categories_new = ColumnDataSource(data=dict(cats=cell_category_new))
    # TODO: create list that contains the neurons the slide can go over, mapping slider index (1 to N) to neuron index
    #       in source.  Depending on dropdown setting, re-make this list to include only accepted, only rejected, all,
    #       or modified-only components.
    neurons_to_show = ColumnDataSource(data=dict(idx=[i for i in range(len(cell_category_original))]))
    print(len(categories.data['cats']))
    print(len(categories_new.data['cats']))
    print(len(neurons_to_show.data['idx']))
    code = """
            console.log("Very first line of JS");
            var data = source.data
            var data_ = source_.data
            var indices = neurons_to_show.data['idx'];  // map neuron indices (in source) to slider values
            var f = cb_obj.value - 1  // slider value. (converted to 0-indexing)
            var x = data['x']
            var y = data['y']
            var y2 = data['y2']
            // update source (i.e. single neuron trace) to the currently selected neuron
            for (var i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+indices[f]*x.length]
                y2[i] = data_['z2'][i+indices[f]*x.length]
            }
            // update category
            var cats = categories.data["cats"];
            var cats_new = categories_new.data["cats"];

            var data2_ = source2_.data;
            var data2 = source2.data;
            var c1 = data2['c1'];
            var c2 = data2['c2'];
            var cc1 = data2_['cc1'];
            var cc2 = data2_['cc2'];
            for (var i = 0; i < c1.length; i++) {
                   c1[i] = cc1[indices[f]][i]
                   c2[i] = cc2[indices[f]][i]
            }
            console.log(cats[f]);
            console.log(cats[f] > 0);
            console.log(indices[f]);
            // update button text and color
            btn_idx.label= "#" + String(indices[f]+1);  // Keep 1-indexing for showing neurons
            if (cats[indices[f]] > 0) {
                btn_orig_cat.label = 'Original: accepted';
                btn_orig_cat.background = 'green';
            }
            else {
                btn_orig_cat.label = 'Original: rejected';
                btn_orig_cat.background = 'red';
            }
            if (cats_new[indices[f]] > 0) {
                btn_new_cat.label = 'Current: accepted';
                btn_new_cat.background = 'green';
            }
            else {
                btn_new_cat.label = 'Current: rejected';
                btn_new_cat.background = 'red';
            }
            source2.change.emit();
            source.change.emit();
        """

    if r_values is not None:
        code += """
            var mets = metrics.data['mets']
            mets[1] = metrics_.data['R'][indices[f]].toFixed(3)
            mets[2] = metrics_.data['SNR'][indices[f]].toFixed(3)
            metrics.change.emit();
        """
        metrics = ColumnDataSource(data=dict(y=(3, 2, 1, 0),
                                             mets=('', "% 7.3f" % r_values[0], "% 7.3f" % SNR[0],
                                                   "N/A" if np.sum(cnn_preds) in (0, None) else "% 7.3f" % cnn_preds[
                                                       0]),
                                             keys=("Evaluation Metrics", "Spatial corr:", "SNR:", "CNN:")))
        if np.sum(cnn_preds) in (0, None):
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR))
        else:
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR, CNN=cnn_preds))
            code += """
                mets[3] = metrics_.data['CNN'][indices[f]].toFixed(3)
            """
        labels = LabelSet(x=0, y='y', text='keys', source=metrics, render_mode='canvas')
        labels2 = LabelSet(x=10, y='y', text='mets', source=metrics, render_mode='canvas', text_align="right")
        plot2 = bpl.figure(plot_width=200, plot_height=100, toolbar_location=None)
        plot2.axis.visible = False
        plot2.grid.visible = False
        plot2.tools.visible = False
        plot2.line([0, 10], [0, 4], line_alpha=0)
        plot2.add_layout(labels)
        plot2.add_layout(labels2)
    else:
        metrics, metrics_ = None, None
    btn_idx = Button(label="#" + str(neurons_to_show.data['idx'][0]+1), disabled=True, width=60)
    original_status = Button(label="original: accepted" if cell_category_original[0] > 0 else "original: rejected",
                             disabled=True, width=150, background="green" if cell_category_original[0] > 0 else "red")
    current_status = Button(label="current: accepted" if cell_category_new[0] > 0 else "current: rejected",
                            disabled=True, width=150, background="green" if cell_category_new[0] > 0 else "red")

    callback = CustomJS(args=dict(source=source, source_=source_,
                                  source2=source2, source2_=source2_,
                                  metrics=metrics, metrics_=metrics_,
                                  categories=categories,
                                  categories_new=categories_new,
                                  btn_idx=btn_idx,
                                  btn_orig_cat=original_status,
                                  btn_new_cat=current_status,
                                  neurons_to_show=neurons_to_show), code=code)

    plot = bpl.figure(plot_width=600, plot_height=200, x_range=Range1d(0, Y_r.shape[1]))
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1,
                  line_alpha=0.6, color=denoised_color)

    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr,
                       plot_width=int(min(1, d2 / d1) * 300),
                       plot_height=int(min(1, d1 / d2) * 300))

    plot1.image(image=[image_neurons[::-1, :]], x=0,
                y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1', 'c2', alpha=0.6, color='purple',
                line_width=2, source=source2)

    transfer_button = Button(label="Transfer", width=100)
    save_button = Button(label="Save changes", width=100)

    menu = [("Rejected", "rejected"), ("Accepted", "accepted"), ("All", "all"), ("Modified", "modified")]
    dropdown = Dropdown(label="Show " + menu[cell_category_original[2]][1], button_type="warning", menu=menu, width=100,
                        name="dropdown")  # default is to show all

    slider = Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                    title="Neuron Number")

    if not (Y_r.shape[0] > 1):
        # bpl.show(row(plot1 if r_values is None else
        #              column(plot1, plot2), plot))
        raise NotImplementedError("Y_r.shape[0] !> 1. This case has not been implemented yet.")

    #slider.js_on_change('value', callback)

    dropdown.js_on_event("menu_item_click", CustomJS(
        args=dict(dropdown=dropdown, slider=slider, categories=categories, categories_new=categories_new,
                  neurons_to_show=neurons_to_show), code=
        """
        var cats_orig = categories.data['cats'];
        // current dropdown selection is item
        // Change label
        dropdown.label = 'Show ' + this.item;
        // Change slider values
        if (this.item == 'accepted') { // show originally accepted
            const n_accepted = categories.data['cats'].reduce((a, b) => a + b, 0);
            // Create an empty array for the indices of accepted components. array[i] = index of i-th accepted neuron.
            var accepted_indices = [];
            accepted_indices.length = n_accepted; 
            accepted_indices.fill(0);
            // TODO: need to get list of indices in categories that are non-zero. Iterate through categories,
            // if element is non-zero, change next element in accepted_indices to the value. Increment accepted_indices pointer.
            // If this kind of rebuilding is too slow, can create more data sources, and change them every time we change neuron classification.
            var i_current = 0; // pointer to first  empty position in accepted_indices 
            for (var i = 0; i < cats_orig.length; i++) {
                if (cats_orig[i] > 0) { // the component was accepted originally 
                    accepted_indices[i_current] = i;
                    i_current += 1; 
                }
            }
            console.log("Show accepted");
            console.log(neurons_to_show.data['idx'].length);
            neurons_to_show.data['idx'] = accepted_indices;
            console.log(neurons_to_show.data['idx'].length);
            slider.end = neurons_to_show.data['idx'].length;
        }
        else if (this.item == 'rejected') {
            const n_rejected = cats_orig.length - categories.data['cats'].reduce((a, b) => a + b, 0);
            
            var rejected_indices = [];
            rejected_indices.length = n_rejected; 
            rejected_indices.fill(0);
            var i_current = 0; // pointer to first  empty position in rejected_indices 
            for (var i = 0; i < cats_orig.length; i++) {
                if (cats_orig[i] == 0) { // the component was rejected originally 
                    rejected_indices[i_current] = i;
                    i_current += 1; 
                }
            }
            console.log("Show rejected");
            console.log(neurons_to_show.data['idx'].length);
            neurons_to_show.data['idx'] = rejected_indices;
            console.log(neurons_to_show.data['idx'].length);
            slider.end = neurons_to_show.data['idx'].length;
        }
        else if (this.item == 'modified') {
            var cats_new = categories_new.data['cats'];
            // TODO: get number of cats_orig = cats_new, then get those components.
            //TODO: do not look at cat_new but the temporary value that will be saved to file.
            var n_modified = 0;
            var modified_indices = [];
            for (var i = 0; i < cats_orig.length; i++){
                if (cats_orig[i] != cats_new[i]) {
                    n_modified++;
                    modified_indices.push(i);
                }
            }
            if (n_modified > 0){
                console.log("Show modified");
                console.log(neurons_to_show.data['idx'].length);
                neurons_to_show.data['idx'] = modified_indices;
                console.log(neurons_to_show.data['idx'].length);
                slider.end = neurons_to_show.data['idx'].length;
                }
        }
        else { // show all components
            var all_indices = [];
            all_indices.length = cats_orig.length; 
            all_indices.fill(0);  // TODO: probably possible to replace loop below with function in fill()
            for (var i = 0; i < cats_orig.length; i++) {
                all_indices[i] = i;
            }
            console.log("Show all");
            console.log(neurons_to_show.data['idx'].length);
            neurons_to_show.data['idx'] = all_indices;
            console.log(neurons_to_show.data['idx'].length);
            slider.end = neurons_to_show.data['idx'].length;
        }
        
    
        """))

    # on pressing transfer, change the current category of the neuron.
    on_transfer_pressed = CustomJS(
        args={'transfer_button': transfer_button, 'curr_cat': categories_new, 'btn_curr_cat': current_status,
              'slider': slider}, code="""
    var i_cell = slider.value - 1
    var cats_new = curr_cat.data['cats'];
    // change current category
    if (cats_new[i_cell] > 0) { // currently accepted -> change to rejected
        cats_new[i_cell] = 0;
        btn_curr_cat.label = 'current: rejected';
        btn_curr_cat.background = 'red';
    }
    else {  // currently rejected -> change to accepted
        cats_new[i_cell] = 1;
        btn_curr_cat.label = 'current: accepted';
        btn_curr_cat.background = 'green';
    }
    // cats_new.change.emit();  //change is undefined here
    """)

    out_fname = get_filename_with_date("manual_classification", extension='.txt')
    save_data_callback = CustomJS(
        args={'new_cats': categories_new, 'out_fname': out_fname},
        code=
        """
        var data = new_cats.data['cats'];
        var out = "";
        for (var i=0; i < data.length; i++) {
            out += data[i];
            out += " ";
        }
        var file = new Blob([out], {type: 'text/plain'});
        var elem = window.document.createElement('a');
        elem.href = window.URL.createObjectURL(file);
        elem.download = out_fname;
        document.body.appendChild(elem);
        elem.click();
        document.body.removeChild(elem);
        """
    )

    transfer_button.js_on_click(on_transfer_pressed)
    save_button.js_on_click(save_data_callback)
    bpl.show(layout([[slider, transfer_button, btn_idx, original_status, current_status, dropdown, save_button],
                     row(plot1 if r_values is None else column(plot1, plot2), plot)]))
    # return Y_r

    # TODO: create save button to write results to a txt file. See https://stackoverflow.com/questions/54215667/bokeh-click-button-to-save-widget-values-to-txt-file-using-javascript
    # and https://stackoverflow.com/questions/62290866/python-bokeh-applicationunable-to-export-updated-data-from-webapp-to-local-syst
    return out_fname


def nb_view_components_with_lfp_movement(estimates,
                                         t_lfp: np.array = None, y_lfp: np.array = None,
                                         t_mov: np.array = None, y_mov: np.array = None,
                                         Yr=None, img=None, idx=None, denoised_color=None, cmap='jet', thr=0.99):
    """view spatial and temporal components interactively in a notebook, along with LFP and movement

    Args:
        estimates : the estimates attribute of a CNMF instance
        t_lfp: np.ndarray
            time data of lfp recording
        y_lfp: np.ndarray
            amplitude of lfp recording
        t_mov: np.ndarray
            time data of movement recording
        y_mov: np.ndarray
            amplitude of movement recording
        Yr :    np.ndarray
            movie in format pixels (d) x frames (T)

        img :   np.ndarray
            background image for contour plotting. Default is the mean
            image of all spatial components (d1 x d2)

        idx :   list
            list of components to be plotted

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param estimates:
    """
    from matplotlib import pyplot as plt
    import scipy

    if 'csc_matrix' not in str(type(estimates.A)):
        estimates.A = scipy.sparse.csc_matrix(estimates.A)

    plt.ion()
    nr, T = estimates.C.shape
    if estimates.R is None:
        estimates.R = estimates.YrA
    if estimates.R.shape != [nr, T]:
        if estimates.YrA is None:
            estimates.compute_residuals(Yr)
        else:
            estimates.R = estimates.YrA

    if img is None:
        img = np.reshape(np.array(estimates.A.mean(axis=1)), estimates.dims, order='F')

    if idx is None:
        nb_view_patches_with_lfp_movement(
            Yr, estimates.A, estimates.C, estimates.b, estimates.f, estimates.dims[0], estimates.dims[1],
            t_lfp=t_lfp, y_lfp=y_lfp, t_mov=t_mov, y_mov=y_mov,
            YrA=estimates.R, image_neurons=img, thr=thr, denoised_color=denoised_color, cmap=cmap,
            r_values=estimates.r_values, SNR=estimates.SNR_comp, cnn_preds=estimates.cnn_preds)
    else:
        nb_view_patches_with_lfp_movement(
            Yr, estimates.A.tocsc()[:, idx], estimates.C[idx], estimates.b, estimates.f,
            estimates.dims[0], estimates.dims[1], t_lfp=t_lfp, y_lfp=y_lfp, t_mov=t_mov, y_mov=y_mov,
            YrA=estimates.R[idx], image_neurons=img,
            thr=thr, denoised_color=denoised_color, cmap=cmap,
            r_values=None if estimates.r_values is None else estimates.r_values[idx],
            SNR=None if estimates.SNR_comp is None else estimates.SNR_comp[idx],
            cnn_preds=None if np.sum(estimates.cnn_preds) in (0, None) else estimates.cnn_preds[idx])
    return estimates


def nb_view_components_manual_control(estimates,
                                      Yr=None, img=None, idx=None, denoised_color=None, cmap='jet', thr=0.99,
                                      mode: str = "all"):
    """view spatial and temporal components interactively in a notebook

    Args:
        estimates : the estimates attribute of a CNMF instance
        t_lfp: np.ndarray
            time data of lfp recording
        y_lfp: np.ndarray
            amplitude of lfp recording
        t_mov: np.ndarray
            time data of movement recording
        y_mov: np.ndarray
            amplitude of movement recording
        Yr :    np.ndarray
            movie in format pixels (d) x frames (T)

        img :   np.ndarray
            background image for contour plotting. Default is the mean
            image of all spatial components (d1 x d2)

        idx :   list
            list of components to be plotted

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param estimates:

		mode: string, one of ["all", "rejected", "accepted"]  # "modified" is also a category but it would be empty
			Whether to go through accepted components and reject manually ("accepted" or reject"), or go through rejected components and move manually to accepted ("rejected" or "accept").
    """
    from matplotlib import pyplot as plt
    import scipy
    # TODO: if refit is used, estimates.idx_components and idx_components_bad are empty (None). Need to still plot
    #  these as all accepted
    if 'csc_matrix' not in str(type(estimates.A)):
        estimates.A = scipy.sparse.csc_matrix(estimates.A)

    if hasattr(estimates, "idx_components"):
        if estimates.idx_components is not None:
            idx_accepted = estimates.idx_components
    else:
        raise Exception("estimates does not have idx_components field")
    if hasattr(estimates, "idx_components_bad"):
        if estimates.idx_components_bad is not None:
            idx_rejected = estimates.idx_components_bad
    else:
        raise Exception("estimates does not have idx_components_bad field")
    plt.ion()
    nr, T = estimates.C.shape
    if estimates.R is None:
        estimates.R = estimates.YrA
    if estimates.R.shape != [nr, T]:
        if estimates.YrA is None:
            estimates.compute_residuals(Yr)
        else:
            estimates.R = estimates.YrA

    if img is None:
        img = np.reshape(np.array(estimates.A.mean(axis=1)), estimates.dims, order='F')

    out_fname = nb_view_patches_manual_control(
        Yr, estimates.A.tocsc(), estimates.C, estimates.b, estimates.f,
        estimates.dims[0], estimates.dims[1],
        YrA=estimates.R, image_neurons=img,
        thr=thr, denoised_color=denoised_color, cmap=cmap,
        r_values=None if estimates.r_values is None else estimates.r_values,
        SNR=None if estimates.SNR_comp is None else estimates.SNR_comp,
        cnn_preds=None if np.sum(estimates.cnn_preds) in (0, None) else estimates.cnn_preds,
        mode=mode,
        idx_accepted=idx_accepted,
        idx_rejected=idx_rejected)
    # return estimates
    return out_fname


def reopen_manual_control(fname: str) -> List:
    """
    :param fname: the file name parameter of nb_view_components_manual_control
    :return: list where each element is 0 or 1, corresponding to whether neuron i is rejected (0) or accepted (1) after
            manual inspection.
    """
    from labrotation.file_handling import open_dir
    import os
    downloads_folder = open_dir("Find downloads directory.")
    file_string = ""
    with open(os.path.join(downloads_folder, fname), "r") as f:
        for line in f.readlines():
            file_string += line
    return list(map(lambda s: int(s.rstrip()), file_string.rstrip().split(" ")))
