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
        else:
            raise ModuleNotFoundError("matlab-2p path was not ")
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
            # TODO: nikon_movie should be closed properly upon removing this class (or does the garbage collector take care of it?)
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

    def _lfp_movement_raw(self):
        self.lfp_file.setSweep(sweepNumber=0, channel=1)
        return (self.lfp_file.sweepX, self.lfp_file.sweepY)

    def _lfp_lfp_raw(self):
        self.lfp_file.setSweep(sweepNumber=0, channel=0)
        return (self.lfp_file.sweepX, self.lfp_file.sweepY)


    def lfp_movement(self):
        """
        Returns columns of the lfp data internal dataframe as pandas Series: a tuple of (t_series, y_series) for lfp
        channel 0 (lfp) data.
        :return: tuple of two pandas Series
        """
        return (self.lfp_df["t_mov_corrected"], self.lfp_df["y_mov"])


    def lfp_lfp(self):
        """
        Returns columns of the lfp data internal dataframe as pandas Series: a tuple of (t_series, y_series) for lfp
        channel 1 (movement) data.
        :return: tuple of two pandas Series
        """
        return (self.lfp_df["t_lfp_corrected"], self.lfp_df["y_lfp"])


    def labview_movement(self):
        """
        Returns columns of the labview data internal dataframe as pandas Series: a tuple of (t_series, y_series)
        for labView time and speed data.
        :return: tuple of two pandas Series
        """
        return (self.belt_df.time_s, self.belt_df.speed)


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
        match_type: "Nikon" or "zero". "Nikon": match (cut) to the Nikon frames (NIDAQ time stamps). "zero": match to 0 s, and the last Nikon frame.
        """
        cut_begin = 0.0
        cut_end = self.nikon_daq_time.iloc[-1]
        if match_type == "Nikon":  # TODO: match_type does not explain its own function
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
        t_mov, y_mov = self._lfp_movement_raw()
        t_lfp, y_lfp = self._lfp_lfp_raw()
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
        # TODO: document columns of dataframes. corrected vs offset
        lfp_df_new["t_lfp_corrected"] = lfp_df_new["t_lfp_offset"] * LFP_SCALING_FACTOR
        lfp_df_new["y_lfp"] = y_lfp

        # cut lfp data
        # cut to Nikon. LFP will not start at 0!
        # lfp_df_new_cut = self._create_lfp_cut_df(
        #    lfp_df_new, self.nikon_daq_time.iloc[0], self.nikon_daq_time.iloc[-1])
        lfp_df_new_cut = self._create_lfp_cut_df(
            lfp_df_new, cut_begin, cut_end)

        return lfp_df_new, lfp_df_new_cut

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

        params["time_offs"] = self.time_offs  # TODO: time_offs does not explain what it is (time offset between LFP and Nikon?)
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


def open_session(data_path: str) -> TwoPhotonSession:
    # FIXME: askopenfilename from TKinter not working. Put into file_handling?
    # TODO: test this function! Make it an alternative constructor? Or a static method
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
