import labrotation.file_handling as fh
import os
import pandas as pd

DATADOC_FOLDER = None
GROUPING_DF = None  # df containing files belonging together in a session
SEGMENTATION_DF = None  # df containing segmentation

def setDocumentationFolder(datadoc_folder: str = None):
    global DATADOC_FOLDER
    if datadoc_folder is None:
        DATADOC_FOLDER = fh.open_dir()
    else:
        DATADOC_FOLDER = datadoc_folder
def loadDataDoc():
    global GROUPING_DF, SEGMENTATION_DF
    # reset the dataframes
    GROUPING_DF = None
    SEGMENTATION_DF = None
    for root, dirs, files in os.walk(DATADOC_FOLDER):
        for name in files:
            if "grouping" in name:
                if "~" in name:  # "~" on windows is used for temporary files that are opened in excel
                    raise Exception(
                        f"Please close all excel files and try again. Found temporary file in:\n{os.path.join(root, name)}")
                else:
                    df = pd.read_excel(os.path.join(root, name))
                    if GROUPING_DF is None:
                        GROUPING_DF = df
                    else:
                        GROUPING_DF = pd.concat(GROUPING_DF, df)
            elif "segmentation" in name:
                if "~" in name:  # "~" on windows is used for temporary files that are opened in excel
                    raise Exception(
                        f"Please close all excel files and try again. Found temporary file in:\n{os.path.join(root, name)}")
                else:
                    df = pd.read_excel(os.path.join(root, name))
                    if SEGMENTATION_DF is None:
                        SEGMENTATION_DF = df
                    else:
                        SEGMENTATION_DF = pd.concat(SEGMENTATION_DF, df)

def getUUIDForFileDeprecated(fpath):
    if os.path.splitext(fpath)[-1] == ".nd2":
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

def getUUIDForFile(fpath):
    fname = os.path.split(fpath)[-1]
    global GROUPING_DF
    df = GROUPING_DF[GROUPING_DF["nd2"] == fname]
    return df["uuid"]



def getSessionFiles():
    pass