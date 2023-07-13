import labrotation.file_handling as fh
import os
import pandas as pd
from collections.abc import Iterable

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
    WIN_INJ_TYPES_DF = None  # df containing window side, type, injection side, type.
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
        if self.WIN_INJ_TYPES_DF is None:
            raise Exception(f"Error: window_injection_types_sides.xlsx was not found in data documentation! \
            Possible reason is the changed structure of data documentation. This file was moved out of 'documentation'. Do not move it back!")

    def getIdUuid(self):
        if self.GROUPING_DF is not None:
            return self.GROUPING_DF[["mouse_id", "uuid"]]
        else:
            raise Exception(
                "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first to populate DataDocumentation object")

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
            return self.GROUPING_DF[self.GROUPING_DF["uuid"] == uuid].nd2.values[0]
        else:
            raise Exception(
                "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first to populate "
                "DataDocumentation object")
    def getSessionFilesForUuuid(self, uuid):
        if self.GROUPING_DF is not None:
            return self.GROUPING_DF[self.GROUPING_DF["uuid"] == uuid]
        else:
            raise Exception(
                "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first to populate "
                "DataDocumentation object")