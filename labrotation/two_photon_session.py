import os
from tkinter.filedialog import askopenfilename
from typing import Optional

import labrotation.belt_processing as belt_processing
import pyabf as abf  # https://pypi.org/project/pyabf/
import pims_nd2
import pandas as pd
import datetime
import pytz  # for timezones
import numpy as np
import warnings
from copy import deepcopy
import json
import labrotation.nikon_ts_reader as ntsr

# heuristic value, hopefully valid for all recordings made with the digitizer module
LFP_SCALING_FACTOR = 1.0038
DATETIME_FORMAT = "%Y.%m.%d-%H:%M:%S.%f"  # used for saving time stamps. See
# https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
# for further information. %f is 0-padded microseconds to 6 digits.

# TODO: fix save-open mismatch in lfp_t_start and nik_t_start, probably due to UTC setting differences
# TODO IMPORTANT: use os.path for path-related manipulations instead of simple string indexing! (like in export_json)
# TODO: make export_json more verbose! (print save directory and file name, for example)
# TODO: replace [:-4] or [:-*] with string manipulation
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
class TwoPhotonSession:
    """
    Attributes:
        Basic:
            ND2_PATH: str
            ND2_TIMESTAMPS_PATH: str
            LABVIEW_PATH: str
            LABVIEW_TIMESTAMPS_PATH: str
            LFP_PATH: str
            MATLAB_2p_FOLDER: str

        Implied (assigned using basic attributes)
            nikon_movie
            nikon_meta
            lfp_file
            belt_dict
            belt_scn_dict
            belt_df
            belt_scn_df
            nikon_daq_time
            lfp_df
            lfp_df_cut
            time_offs_lfp_nik
            belt_params
            lfp_t_start
            nik_t_start
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
            self.MATLAB_2p_FOLDER = matlab_2p_folder
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
        for k, v in instance.belt_dict.items():  # convert matlab arrays into numpy arrays
            instance.belt_dict[k] = instance._matlab_array_to_numpy_array(
                instance.belt_dict[k])
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
        with open(fpath, "r") as json_file:
            json_dict = json.load(json_file)
            # get basic attributes
            # TODO: handle missing basic attributes as error
            nd2_path = json_dict["ND2_PATH"]
            nd2_timestamps_path = json_dict["ND2_TIMESTAMPS_PATH"]
            labview_path = json_dict["LABVIEW_PATH"]
            labview_timestamps_path = json_dict["LABVIEW_TIMESTAMPS_PATH"]
            lfp_path = json_dict["LFP_PATH"]
            matlab_2p_folder = json_dict["MATLAB_2p_FOLDER"]
        # optional inferred attributes are in json_dict, few of them need modification
        # lfp_t_start has in format .strftime("%Y.%m.%d-%H:%M:%S")
        json_dict["lfp_t_start"] = datetime.datetime.strptime(json_dict["lfp_t_start"], DATETIME_FORMAT)
        # nik_t_start has format .strftime("%Y.%m.%d-%H:%M:%S")
        json_dict["nik_t_start"] = datetime.datetime.strptime(json_dict["nik_t_start"], DATETIME_FORMAT)
        # time_offs_lfp_nik is float, not str
        json_dict["time_offs_lfp_nik"] = float(json_dict["time_offs_lfp_nik"])
        # lfp_scaling is float, not str
        json_dict["lfp_scaling"] = float(json_dict["lfp_scaling"])
        return cls(nd2_path=nd2_path,
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

    def _open_data(self):
        if self.ND2_PATH is not None:
            self.nikon_movie = pims_nd2.ND2_Reader(self.ND2_PATH)
            # TODO: nikon_movie should be closed properly upon removing this class (or does the garbage collector
            #  take care of it?)
        if self.ND2_TIMESTAMPS_PATH is not None:
            try:
                self.nikon_meta = pd.read_csv(
                    self.ND2_TIMESTAMPS_PATH, delimiter="\t", encoding="utf_16_le")
            except UnicodeDecodeError:
                print(
                    "_open_data(): Timestamp file seems to be unusual. Trying to correct it.")
                output_file_path = self.ND2_TIMESTAMPS_PATH[:-4] + "_corrected.txt"
                ntsr.standardize_stamp_file(
                    self.ND2_TIMESTAMPS_PATH, output_file_path, export_encoding="utf_16_le")
                self.nikon_meta = pd.read_csv(
                    output_file_path, delimiter="\t", encoding="utf_16_le")
                self.ND2_TIMESTAMPS_PATH = output_file_path
        if hasattr(self, "LABVIEW_PATH") and self.LABVIEW_PATH is not None:
            self.belt_dict, self.belt_scn_dict, self.belt_params = belt_processing.beltProcessPipelineExpProps(
                self.LABVIEW_PATH, self.ND2_TIMESTAMPS_PATH, self.MATLAB_2p_FOLDER)
        if hasattr(self, "LFP_PATH") and self.LFP_PATH is not None:
            self.lfp_file = abf.ABF(self.LFP_PATH)
            self.lfp_scaling = LFP_SCALING_FACTOR

    def _matlab_array_to_numpy_array(self, matlab_array):
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
                self.MATLAB_2p_FOLDER
            Inferred attributes:
                (nikon_movie)
                (nikon_meta)
                (lfp_file)
                (belt_dict)
                (belt_scn_dict)
                (belt_df)
                (belt_scn_df)
                (nikon_daq_time)
                (lfp_df)
                (lfp_df_cut)
                self.time_offs_lfp_nik
                (belt_params)
                self.lfp_t_start
                self.nik_t_start
                self.lfp_scaling

                belt_params (dict)
        """
        fpath = kwargs.get("fpath", self.ND2_PATH[:-3] + "json")

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
        params["MATLAB_2p_FOLDER"] = self.MATLAB_2p_FOLDER

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

    # TODO: get nikon frame matching time stamps (NIDAQ time)! It is session.nikon_daq_time
    def return_nikon_mean(self):
        return [self.nikon_movie[i_frame].mean() for i_frame in range(self.nikon_movie.sizes["t"])]


def open_session(data_path: str) -> TwoPhotonSession:
    # FIXME: askopenfilename from TKinter not working. Put into file_handling?
    # TODO: test this function! Make it an alternative constructor
    # .nd2 file
    nd2_path = askopenfilename(initialdir=data_path, title="Select .nd2 file")
    print(f"Selected imaging file: {nd2_path}")

    # nd2 info file (..._nik.txt) Image Proterties -> Recorded Data of .nd2 file saved as .txt
    nd2_timestamps_path = nd2_path[:-4] + "_nik" + ".txt"
    if not os.path.exists(nd2_timestamps_path):
        nd2_timestamps_path = askopenfilename(initialdir=data_path,
                                              title="Nikon info file not found. Please provide it!")
    print(f"Selected nd2 info file: {nd2_timestamps_path}")

    # labview .txt file
    labview_path = askopenfilename(
        initialdir=data_path, title="Select corresponding labview (xy.txt) file")
    print(f"Selected LabView data file: {labview_path}")

    # labview time stamp (...time.txt)
    labview_timestamps_path = labview_path[
                              :-4] + "time" + ".txt"  # try to open the standard corresponding time stamp file first
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
