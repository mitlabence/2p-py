import labrotation.file_handling as fh
import os
import pandas as pd
from collections.abc import Iterable
import numpy as np

# TODO: a module is less complicated, but setting
#   %load_ext autoreload
#   %autoreload 2
# in jupyter causes module to reload, constantly resetting the folders.

class DataDocumentation:
    """
    SEGMENTS_CNMF_CATS and SEGMENTS_MOCO_CATS assign to each category appearing in the data documentation
    (segmentation) the boolean value whether the CNMF and MoCo can or should run on segments belonging in that category.
    These categories should be exactly the unique categories appearing in the [mouse-id]_segmentation.xlsx files, or,
    once SEGMENTATION_DF contains all this data, in SEGMENTATION_DF["interval_type"].unique().
    """
    DATADOC_FOLDER = None
    GROUPING_DF = None  # df containing files belonging together in a session
    SEGMENTATION_DF = None  # df containing segmentation
    COLORINGS_DF = None  # df containing color code for each mouse ID
    WIN_INJ_TYPES_DF = None  # df containing window side, type, injection side, type.
    EVENTS_DF = None  # df containing all events and the corresponding metadata
    SEGMENTS_CNMF_CATS = {"normal": True, "iis": False, "sz": False, "sz_like": False, "sd_wave": False,
                          "sd_extinction": False,
                          "fake_handling": False, "sd_wave_delayed": False, "sd_extinction_delayed": False,
                          "stimulation": False, "sd_wave_cx": False}
    SEGMENTS_MOCO_CATS = {"normal": True, "iis": True, "sz": True, "sz_like": True, "sd_wave": True,
                          "sd_extinction": True,
                          "fake_handling": True, "sd_wave_delayed": True, "sd_extinction_delayed": True,
                          "stimulation": False, "sd_wave_cx": True}

    def __init__(self, datadoc_folder: str = None):
        if datadoc_folder is None:
            self.DATADOC_FOLDER = fh.open_dir("Open data documentation")
        else:
            self.DATADOC_FOLDER = datadoc_folder

    def checkCategoryConsistency(self):
        n_segments = len(self.SEGMENTATION_DF["interval_type"].unique())
        n_segments_cnmf = len(self.SEGMENTS_CNMF_CATS.keys())
        n_segments_moco = len(self.SEGMENTS_MOCO_CATS.keys())
        if n_segments != n_segments_cnmf:
            raise Exception(
                f"Found {n_segments} segment types in data documentation: \
                {self.SEGMENTATION_DF['interval_type'].unique()} vs {n_segments_cnmf} defined in datadoc_util.py\
                (SEGMENTS_CNMF_CATS): {self.SEGMENTS_CNMF_CATS.keys()}")
        if n_segments != n_segments_moco:
            raise Exception(
                f"Found {n_segments} segment types in data documentation: \
                           {self.SEGMENTATION_DF['interval_type'].unique()} vs {n_segments_cnmf} defined in datadoc_util.py\
                           (SEGMENTS_MOCO_CATS): {self.SEGMENTS_MOCO_CATS.keys()}")
        print("DataDocumentation.checkCategoryConsistency(): Categories seem consistent.")

    def loadDataDocTest(self):
        for root, dirs, files in os.walk(self.DATADOC_FOLDER):
            for name in files:
                if "grouping" in name:
                    if "~" in name:  # "~" on windows is used for temporary files that are opened in excel
                        print(os.path.join(root, name))
                elif "segmentation" in name:
                    if "~" in name:  # "~" on windows is used for temporary files that are opened in excel
                        print(os.path.join(root, name))
    def checkFileConsistency(self, ):
        """
        Go over the sessions contained in the data documentation; print the files that were not found.
        :return:
        """
        count_not_found = 0
        for i_row, grouping_row in self.GROUPING_DF.iterrows():
            folder = grouping_row["folder"]
            nd2_fname = grouping_row["nd2"]
            lv_fname = grouping_row["labview"]
            lfp_fname = grouping_row["lfp"]
            facecam_fname = grouping_row["face_cam_last"]
            nikonmeta_fname = grouping_row["nikon_meta"]
            if isinstance(nd2_fname, str):
                fpath_complete = os.path.join(folder, nd2_fname)
                if not os.path.exists(fpath_complete):
                    print(f"Could not find {fpath_complete}")
                    count_not_found += 1
            if isinstance(lv_fname, str):
                fpath_complete = os.path.join(folder, lv_fname)
                if not os.path.exists(fpath_complete):
                    print(f"Could not find {fpath_complete}")
                    count_not_found += 1
            if isinstance(lfp_fname, str):
                fpath_complete = os.path.join(folder, lfp_fname)
                if not os.path.exists(fpath_complete):
                    print(f"Could not find {fpath_complete}")
                    count_not_found += 1
            if isinstance(facecam_fname, str):
                fpath_complete = os.path.join(folder, facecam_fname)
                if not os.path.exists(fpath_complete):
                    print(f"Could not find {fpath_complete}")
                    count_not_found += 1
            if isinstance(nikonmeta_fname, str):
                fpath_complete = os.path.join(folder, nikonmeta_fname)
                if not os.path.exists(fpath_complete):
                    print(f"Could not find {fpath_complete}")
                    count_not_found += 1
        print(f"Total: {count_not_found} missing files.")

    def loadDataDoc(self):
        # reset the dataframes
        for root, dirs, files in os.walk(self.DATADOC_FOLDER):
            for name in files:
                if "grouping" in name:
                    if "~" in name:  # "~" on windows is used for temporary files that are opened in excel
                        raise Exception(
                            f"Please close all excel files and try again. Found temporary file in:\n{os.path.join(root, name)}")
                    else:
                        df = pd.read_excel(os.path.join(root, name))
                        mouse_id = os.path.splitext(name)[0].split("_")[
                            0]  # get rid of extension, then split xy_grouping to get xy
                        df["mouse_id"] = mouse_id
                        if self.GROUPING_DF is None:
                            self.GROUPING_DF = df
                        else:
                            self.GROUPING_DF = pd.concat([self.GROUPING_DF, df])
                elif "segmentation" in name:
                    if "~" in name:  # "~" on windows is used for temporary files that are opened in excel
                        raise Exception(
                            f"Please close all excel files and try again. Found temporary file in:\n{os.path.join(root, name)}")
                    else:
                        df = pd.read_excel(os.path.join(root, name))
                        if self.SEGMENTATION_DF is None:
                            self.SEGMENTATION_DF = df
                        else:
                            self.SEGMENTATION_DF = pd.concat([self.SEGMENTATION_DF, df])
                elif name == "window_injection_types_sides.xlsx":
                    self.WIN_INJ_TYPES_DF = pd.read_excel(os.path.join(root, name))
                elif name == "events_list.xlsx":
                    self.EVENTS_DF = pd.read_excel(os.path.join(root, name))
        if self.WIN_INJ_TYPES_DF is None:
            raise Exception(f"Error: window_injection_types_sides.xlsx was not found in data documentation! \
            Possible reason is the changed structure of data documentation. This file was moved out of 'documentation'. Do not move it back!")
        self.COLORINGS_DF = self.getColorings()

    def setDataDriveSymbol(self, symbol: str=None):
        if isinstance(symbol, str):
            assert len(symbol) == 1
            assert symbol.upper() == symbol
            self.GROUPING_DF.folder = self.GROUPING_DF.apply(lambda row: symbol + row["folder"][1:], axis=1)
            print(f"Changed drive symbol to {symbol}")

    def getIdUuid(self):
        if self.GROUPING_DF is not None:
            return self.GROUPING_DF[["mouse_id", "uuid"]]
        else:
            raise Exception(
                "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first to populate DataDocumentation object")

    def getColorings(self):
        """
        Read out the 'color coding.xlsx' of the data documentation, which should contain ID - color hex, r, g, b pairs.
        :return: pandas dataframe
        """
        color_coding_fpath = os.path.join(self.DATADOC_FOLDER, "color coding.xlsx")
        if os.path.exists(color_coding_fpath):
            return pd.read_excel(color_coding_fpath)
        else:
            raise Exception(f"File {color_coding_fpath} does not exist.")

    def getUUIDForFileDeprecated(self, nd2_fname, data_docu_folder):
        if os.path.splitext(nd2_fname)[-1] == ".nd2":
            docu_files_list = []
            session_uuid = None
            for root, dirs, files in os.walk(data_docu_folder):
                for name in files:
                    if "grouping" in name:
                        if "~" in name:  # "~" on windows is used for temporary files that are opened in excel
                            docu_files_list = []
                            raise Exception(
                                f"Please close all excel files and try again. Found temporary file in:\n{os.path.join(root, name)}")
                        fpath = os.path.join(root, name)
                        df = pd.read_excel(fpath)
                        df = df[df["nd2"] == nd2_fname]
                        if len(df) > 0:
                            if len(df) > 1:
                                raise Exception(
                                    f"File name appears several times in data documentation:\n\t{nd2_fname}\n{df}")
                            else:
                                session_uuid = df["uuid"].iloc[0]
                            break
                        docu_files_list.append(fpath)
            if session_uuid is None:
                session_uuid = uuid.uuid4().hex
                warnings.warn(
                    f"Warning: movie does not have entry (uuid) in data documentation!\nYou should add data to documentation. The generated uuid for this session is: {session_uuid}",
                    UserWarning)
            print(f"UUID is {session_uuid}")
            return session_uuid
        else:
            raise NotImplementedError("getUUIDForFile() only implemented for nd2 files so far.")

    def getMouseIdForUuid(self, uuid):  # TODO: handle invalid uuid
        return self.GROUPING_DF[self.GROUPING_DF["uuid"] == uuid].mouse_id.values[0]

    def getMouseWinInjInfo(self, mouse_id):
        return self.WIN_INJ_TYPES_DF[self.WIN_INJ_TYPES_DF["mouse_id"] == mouse_id]

    def getInjectionDirection(self, mouse_id):
        inj_side = self.WIN_INJ_TYPES_DF[self.WIN_INJ_TYPES_DF["mouse_id"] == mouse_id]["injection_side"].values[0]
        win_side = self.WIN_INJ_TYPES_DF[self.WIN_INJ_TYPES_DF["mouse_id"] == mouse_id]["window_side"].values[0]
        top_dir = self.getTopDirection(mouse_id)
        # assert the injection side is opposite to the window side
        if (inj_side == "right" and win_side == "left") or (inj_side == "left" and win_side == "right"):
            # if imaging contralateral to injection, injection is always towards medial side
            return "top" if top_dir == "medial" else "bottom"
        else:  # some mice have window on both hemispheres
            raise NotImplementedError(
                "datadoc_util: getInjectionDirection: only left and right windows, contralateral injections have been implemented.")

    def getUUIDForFile(self, fpath):
        fname = os.path.split(fpath)[-1]
        df = self.GROUPING_DF[self.GROUPING_DF["nd2"] == fname]
        if len(df) > 1:
            raise Exception("nd2 file is not unique!")
        return df["uuid"].iat[0]

    def getSegments(self, nd2_file):
        assert os.path.splitext(nd2_file)[-1] == ".nd2"
        return self.SEGMENTATION_DF[self.SEGMENTATION_DF["nd2"] == nd2_file]

    def getSegmentsForUUID(self, uuid, as_df=True):
        nd2_file = self.GROUPING_DF[self.GROUPING_DF["uuid"] == uuid].nd2.values[0]
        segments_df = self.SEGMENTATION_DF[self.SEGMENTATION_DF["nd2"] == nd2_file]
        segments_df = segments_df.drop("nd2", axis=1)
        if as_df:
            return segments_df
        else:
            ival_type = segments_df["interval_type"].array
            fbegin = segments_df["frame_begin"].array
            fend = segments_df["frame_end"].array
            segments = []
            for i in range(len(ival_type)):
                segments.append((ival_type[i], fbegin[i], fend[i]))
            return segments

    def getSessionFiles(self):
        pass

    def getTopDirection(self, mouse_id):
        # push right button: mouse goes forward, imaging field goes right (cells go left). This means that 
        # right: posterior, left: anterior.
        # Depending on window side:
        # left window: up=towards medial, down: towards lateral
        # right window: up=towards lateral, down: towards medial
        window_side_top_dir_dict = {"left": "medial", "right": "lateral"}
        return window_side_top_dir_dict[self.getMouseWinInjInfo(mouse_id)["window_side"].values[0]]

    def addUUIDColumnFromNd2(self, df):
        """
        Purpose: when returning dataframe, useful to add uuid column instead of only specifying nd2 files.
        This function adds the uuid column to a dataframe (e.g. self.SEGMENTATION_DF)
        :param df:
        :return:
        """
        assert "nd2" in df.columns
        if "uuid" not in df.columns:
            df["uuid"] = df.apply(lambda row: self.GROUPING_DF[self.GROUPING_DF["nd2"] == row["nd2"]].uuid.values[0], axis=1)
        else:
            print("datadoc_util addUUIDColumnFromNd2: uuid column already exists! Returning unchanged dataframe...")
        return df


    def getAllSegmentsWithType(self, segment_type = "normal", experiment_type: str="tmev"):
        """
        Returns all segments in the data documentation that have the defined segment type(s)
        :param segment_type: string or list of strings.
        :param experiment_type: string or list of strings. only take recordings that fall into this category (see data grouping). Example: "tmev", "tmev_bl", "chr2_szsd"
        :return:
        """

        # convert possible string
        if type(segment_type) is str:
            segment_types = [segment_type]
        else:  # assume otherwise list of strings was given
            segment_types = segment_type

        if type(experiment_type) is str:
            experiment_types = [experiment_type]
        else:
            experiment_types = experiment_type
        exptype_unique = self.GROUPING_DF.experiment_type.unique()
        for e_type in experiment_types:  # cannot do for element in experiment_types
            assert e_type in exptype_unique

        res_df = self.addUUIDColumnFromNd2(self.SEGMENTATION_DF[self.SEGMENTATION_DF["interval_type"].isin(segment_types)])
        res_df["experiment_type"] = res_df.apply(lambda row: self.GROUPING_DF[self.GROUPING_DF["nd2"] == row["nd2"]].experiment_type.values[0], axis=1)
        res_df = res_df[res_df["experiment_type"].isin(experiment_types)]
        return res_df #.drop("experiment_type")
    def getNikonFileNameUuid(self):
        if self.GROUPING_DF is not None:
            return self.GROUPING_DF[["nd2", "uuid"]]
        else:
            raise Exception(
                "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first to populate "
                "DataDocumentation object")
    def getNikonFileNameForUuid(self, uuid):
        if self.GROUPING_DF is not None:
            if isinstance(uuid, str):
                return self.GROUPING_DF[self.GROUPING_DF["uuid"] == uuid].nd2.values[0]
            elif isinstance(uuid, list):
                return self.GROUPING_DF[self.GROUPING_DF["uuid"].isin(uuid)].nd2.values
            else:
                raise Exception(f"uuid has type {type(uuid)}; needs to be str or list[str]!")
        else:
            raise Exception(
                "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first to populate "
                "DataDocumentation object")
    def getNikonFilePathForUuid(self, uuid):
        if self.GROUPING_DF is not None:
            if isinstance(uuid, str):
                grouping_entry =  self.GROUPING_DF[self.GROUPING_DF["uuid"] == uuid].iloc[0]
                folder = grouping_entry.folder
                nd2 = grouping_entry.nd2
                return os.path.join(folder, nd2)
            elif isinstance(uuid, list) or isinstance(uuid, np.ndarray):
                fpath_list = []
                for i_entry in range(len(uuid)):
                    entry = self.GROUPING_DF[self.GROUPING_DF["uuid"] == uuid[i_entry]].iloc[0]
                    folder = entry.folder
                    nd2 = entry.nd2
                    fpath_list.append(os.path.join(folder, nd2))
                return fpath_list
            else:
                raise Exception(f"uuid has type {type(uuid)}; needs to be str or list[str]!")
        else:
            raise Exception(
                "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first to populate "
                "DataDocumentation object")
    def getSessionFilesForUuid(self, uuid):
        if self.GROUPING_DF is not None:
            return self.GROUPING_DF[self.GROUPING_DF["uuid"] == uuid]
        else:
            raise Exception(
                "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first to populate "
                "DataDocumentation object")

    def getSegmentForFrame(self, uuid, frame):
        """
        Given a 1-indexed frame, return a row containing interval type, beginning frame, end frame for the segment that the frame belongs to.
        :param uuid: The uuid of the recording.
        :param frame: The 1-indexed frame (i.e. first frame = 1) to get the segment info on.
        :return: a pandas DataFrame with columns "nd2", "interval_type", "frame_begin", "frame_end", and with at most one row, the segment that the frame belongs to.
        """
        nd2_file = self.GROUPING_DF[self.GROUPING_DF["uuid"] == uuid].nd2.values[0]
        return self.SEGMENTATION_DF[(self.SEGMENTATION_DF["nd2"] == nd2_file) & (self.SEGMENTATION_DF["frame_begin"] <= frame) & (self.SEGMENTATION_DF["frame_end"] >= frame)]

    def getColorForMouseId(self, mouse_id):
        if self.COLORINGS_DF is not None:
            if mouse_id in self.COLORINGS_DF.mouse_id.unique():
                return self.COLORINGS_DF[self.COLORINGS_DF["mouse_id"] == mouse_id].iloc[0].color
            else:
                raise Exception(f"Color code for ID {mouse_id} not found")
        else:
            raise Exception("Color codes not yet loaded.")
    def getColorForUuid(self, uuid):
        mouse_id = self.getMouseIdForUuid(uuid)
        color = self.getColorForMouseId(mouse_id)
        return color
    def getRecordingsWithExperimentType(self, experiment_types="fov_dual"):
        """
        Return all grouping info of recordings with the defined experiment_type,
        :param experiment_types: string or list of strings, type(s) of experiment. Some examples: fov_dual, tmev, tmev_bl, chr2_szsd,
        chr2_ctl, chr2_sd
        :return:
        """
        if type(experiment_types) is str:
            return self.GROUPING_DF[self.GROUPING_DF.experiment_type == experiment_types]
        elif type(experiment_types) is list:
            assert type(experiment_types[0]) == str
            return self.GROUPING_DF[self.GROUPING_DF.experiment_type.isin(experiment_types)]
    def getEventsDf(self):
        return self.EVENTS_DF

    def getExperimentTypeForUuid(self, uuid):
        return self.GROUPING_DF[self.GROUPING_DF["uuid"] == uuid].experiment_type.iloc[0]
