import belt_processing
import pyabf as abf  # https://pypi.org/project/pyabf/
import pims_nd2
import pandas as pd
import datetime
import pytz  # for timezones
import numpy as np
import warnings
from copy import deepcopy
import json
import nikon_ts_reader as ntsr
# heuristic value, hopefully valid for all recordings made with the digitizer module
LFP_SCALING_FACTOR = 1.0038


class TwoPhotonSession:
    # TODO: apparently, these are static variables... Bringing them inside __init__ is recommended.
    ND2_PATH: str = None  # TODO: no caps here!
    ND2_TIMESTAMPS_PATH: str = None
    LABVIEW_PATH: str = None
    LABVIEW_TIMESTAMPS_PATH: str = None
    LFP_PATH: str = None
    MATLAB_2p_FOLDER: str = None

    nikon_movie = None
    nikon_meta = None
    lfp_file = None
    belt_dict = None
    belt_scn_dict = None
    belt_df = None
    belt_scn_df = None  # belt data cut to match Nikon frames (same #frames)
    nikon_daq_time = None
    lfp_df = None
    lfp_df_cut = None
    time_offs = None
    belt_params = None  # dictionary containing belt process pipeline parameters
    # IMPORTANT: change the used lfp_scaling to always be the actual scaling used! Important for exporting (saving) the session data for reproducibility
    lfp_scaling: float = LFP_SCALING_FACTOR
    # TODO: we might not have LFP or LabView, consider these cases! All arguments are necessary right now, one solution would be to suppply "" as empty arguments, or test supplying None. Another solution would be optional parameters.

    def __init__(self, nd2_path: str, nd2_timestamps_path: str, labview_path: str, labview_timestamps_path: str,
                 lfp_path: str, matlab_2p_folder: str):
        if nd2_path is not None:
            self.ND2_PATH = nd2_path
        if nd2_timestamps_path is not None:
            self.ND2_TIMESTAMPS_PATH = nd2_timestamps_path
        if labview_path is not None:
            self.LABVIEW_PATH = labview_path
        if labview_timestamps_path is not None:
            self.LABVIEW_TIMESTAMPS_PATH = labview_timestamps_path
        if lfp_path is not None:
            self.LFP_PATH = lfp_path
        if matlab_2p_folder is not None:
            self.MATLAB_2p_FOLDER = matlab_2p_folder
        self._open_data()
        for k, v in self.belt_dict.items():  # convert matlab arrays into numpy arrays
            self.belt_dict[k] = self._matlab_array_to_numpy_array(
                self.belt_dict[k])
        self._nikon_remove_na()
        self._create_nikon_daq_time()
        self.time_offs = self._match_lfp_nikon_stamps()
        self.lfp_df, self.lfp_df_cut = self._create_lfp_df(
            self.time_offs, 0.0, self.nikon_daq_time.iloc[-1])
        self.belt_df = self._create_belt_df(self.belt_dict)
        self.belt_scn_df = self._create_belt_df(self.belt_scn_dict)

    def dirty_close(self):  # TODO: this is not necessary if variables are defined in __init__, not before! (current variables are static because defined above __init__, not in __init__)
        self.ND2_PATH = None  # TODO: no caps here!
        self.ND2_TIMESTAMPS_PATH = None
        self.LABVIEW_PATH = None
        self.LABVIEW_TIMESTAMPS_PATH = None
        self.LFP_PATH = None
        self.MATLAB_2p_FOLDER = None

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
        self.time_offs = None
        self.lfp_t_start = None
        self.nik_t_start = None

    def _open_data(self):
        if self.ND2_PATH is not None:
            self.nikon_movie = pims_nd2.ND2_Reader(self.ND2_PATH)
        if self.ND2_TIMESTAMPS_PATH is not None:
            try:
                self.nikon_meta = pd.read_csv(
                    self.ND2_TIMESTAMPS_PATH, delimiter="\t", encoding="utf_16_le")
            except UnicodeDecodeError:
                print(
                    "_open_data(): Timestamp file seems to be unusual. Trying to correct it.")
                output_file_path = self.ND2_TIMESTAMPS_PATH[:-
                                                            4] + "_corrected.txt"
                ntsr.standardize_stamp_file(
                    self.ND2_TIMESTAMPS_PATH, output_file_path, export_encoding="utf_16_le")
                self.nikon_meta = pd.read_csv(
                    output_file_path, delimiter="\t", encoding="utf_16_le")
                self.ND2_TIMESTAMPS_PATH = output_file_path
        if self.LABVIEW_PATH is not None:
            self.belt_dict, self.belt_scn_dict, self.belt_params = belt_processing.beltProcessPipelineExpProps(
                self.LABVIEW_PATH, self.ND2_TIMESTAMPS_PATH, self.MATLAB_2p_FOLDER)
        if self.LABVIEW_TIMESTAMPS_PATH is not None:
            pass  # TODO: in the future, need this too, probably.
        if self.LFP_PATH is not None:
            self.lfp_file = abf.ABF(self.LFP_PATH)

    def _matlab_array_to_numpy_array(self, matlab_array):
        return np.array(matlab_array._data)

    def _nikon_remove_na(self):
        # get the stimulation metadata frames
        event_times = self.nikon_meta[self.nikon_meta["Index"].isna() == True]
        # drop non-imaging frames from metadata
        self.nikon_meta.dropna(subset=["Index"], inplace=True)

    def lfp_movement(self):
        self.lfp_file.setSweep(sweepNumber=0, channel=1)
        return (self.lfp_file.sweepX, self.lfp_file.sweepY)

    def lfp_lfp(self):
        self.lfp_file.setSweep(sweepNumber=0, channel=0)
        return (self.lfp_file.sweepX, self.lfp_file.sweepY)

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
        Shifts the LFP signal (+/-) by the amount of seconds: an event at time t to time (t+seconds), i.e. a positive seconds means shifting everything to a later time.
        match_type: "Nikon" or "zero". "Nikon": match to the Nikon frames (NIDAQ time stamps). "zero": match to 0 s, and the last Nikon frame.
        """
        cut_begin = 0.0
        cut_end = self.nikon_daq_time.iloc[-1]
        if match_type == "Nikon":
            cut_begin = self.nikon_daq_time.iloc[0]
        self.lfp_df, self.lfp_df_cut = self._create_lfp_df(
            self.time_offs - seconds, cut_begin, cut_end)
        self.time_offs = self.time_offs - seconds

    # TODO: this does not actually matches the two, but gets the offset for matching
    def _match_lfp_nikon_stamps(self) -> float:
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

        time_offset = nik_t_start - lfp_t_start
        time_offset_sec = time_offset.seconds

        time_offs = time_offset_sec + (time_offset.microseconds * 1e-6)

        # stop process if too much time detected between starting LFP and Nikon recording.
        if time_offs > 30.0:
            warnings.warn(
                f"Warning! more than 30 s difference detected between starting the LFP and the Nikon recording!\nPossible cause: bug in conversion to utc (daylight saving mode, timezone conversion).\nlfp: {lfp_t_start}\nnikon: {nik_t_start}\noffset (s): {time_offs}")

        print(f"Difference of starting times (s): {time_offs}")

        return time_offs

    def _create_belt_df(self, belt_dict: dict) -> pd.DataFrame:
        """
        This function takes belt_dict or belt_scn_dict and returns it as a dataframe. Some columns with less entries are removed!
        """
        if "runtime" in belt_dict:  # only reliable way I know of to differentiate between belt and belt_scn.
            bd = belt_dict.copy()
            bd.pop("runtime")
            bd.pop("tsscn")
            df = pd.DataFrame(bd)
        else:
            df = pd.DataFrame(belt_dict)
        if "time" in df.columns:
            df["time_s"] = df["time"]/1000.
        return df

    def _create_lfp_cut_df(self, lfp_df_raw, lower_limit: float, upper_limit: float) -> pd.DataFrame:
        lfp_df_new_cut = pd.DataFrame()
        lfp_df_new_cut = lfp_df_raw[lfp_df_raw["t_lfp_corrected"]
                                    >= lower_limit]
        lfp_df_new_cut = lfp_df_new_cut[lfp_df_new_cut["t_lfp_corrected"]
                                        <= upper_limit]
        return lfp_df_new_cut

    def _create_lfp_df(self, time_offs: float, cut_begin: float, cut_end: float):
        lfp_df_new = pd.DataFrame()
        lfp_df_new_cut = pd.DataFrame()
        t_mov, y_mov = self.lfp_movement()
        t_lfp, y_lfp = self.lfp_lfp()
        # add movement data
        lfp_df_new["t_mov_raw"] = t_mov
        lfp_df_new["t_mov_offset"] = lfp_df_new["t_mov_raw"] - time_offs
        # scale factor given in Bence's excel sheets
        lfp_df_new["t_mov_corrected"] = lfp_df_new["t_mov_offset"] * \
            LFP_SCALING_FACTOR
        lfp_df_new["y_mov"] = y_mov
        # add normalized movement data
        motion_min = lfp_df_new["y_mov"].min()
        motion_max = lfp_df_new["y_mov"].max()
        motion_mean = lfp_df_new["y_mov"].mean()
        # FIXME: normalization should be (y - mean)/(max - min)
        lfp_df_new["y_mov_normalized"] = lfp_df_new["y_mov"]/motion_mean

        # add lfp data
        lfp_df_new["t_lfp_raw"] = t_lfp
        lfp_df_new["t_lfp_offset"] = lfp_df_new["t_lfp_raw"] - time_offs
        # scale factor given in Bence's excel sheets
        lfp_df_new["t_lfp_corrected"] = lfp_df_new["t_lfp_offset"] * \
            LFP_SCALING_FACTOR
        lfp_df_new["y_lfp"] = y_lfp

        # cut lfp data
        # cut to Nikon. LFP will not start at 0!
        # lfp_df_new_cut = self._create_lfp_cut_df(
        #    lfp_df_new, self.nikon_daq_time.iloc[0], self.nikon_daq_time.iloc[-1])
        lfp_df_new_cut = self._create_lfp_cut_df(
            lfp_df_new, cut_begin, cut_end)

        return lfp_df_new, lfp_df_new_cut

    def export_json(self, **kwargs) -> None:
        """
        *args:
            fpath - absolute file path of json file. If not supported, it will be
            nd2 file path and name, with json extension.

        Given an absolute file path fpath to a not existing json file, the important
        parameters of the session are exported into this json file.
        Parameters to serialize:
                self.ND2_PATH
                self.ND2_TIMESTAMPS_PATH
                self.LABVIEW_PATH
                self.LABVIEW_TIMESTAMPS_PATH
                self.LFP_PATH
                self.MATLAB_2p_FOLDER

                self.time_offs
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

        params["nd2_path"] = self.ND2_PATH
        params["nd2_timestamps_path"] = self.ND2_TIMESTAMPS_PATH
        params["labview_path"] = self.LABVIEW_PATH
        params["labview_timestamps_path"] = self.LABVIEW_TIMESTAMPS_PATH
        params["lfp_path"] = self.LFP_PATH
        params["matlab_2p_folder"] = self.MATLAB_2p_FOLDER

        params["time_offs"] = self.time_offs
        params["lfp_t_start"] = self.lfp_t_start.strftime("%Y.%m.%d-%H:%M:%S")
        params["nik_t_start"] = self.nik_t_start.strftime("%Y.%m.%d-%H:%M:%S")
        params["lfp_scaling"] = self.lfp_scaling

        # turn everything into a string for json dump
        for k, v in params.items():
            params[k] = str(v)

        with open(fpath, "w") as f:
            json.dump(params, f, indent=4)

    # TODO: get nikon frame matching time stamps (NIDAQ time)! It is session.nikon_daq_time
    def return_nikon_mean(self):
        return [self.nikon_movie[i_frame].mean() for i_frame in range(self.nikon_movie.sizes["t"])]
