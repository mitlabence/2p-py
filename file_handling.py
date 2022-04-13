from tkinter import Tk  # use tkinter to open files
from tkinter.filedialog import askopenfilename, askdirectory
import os.path
import belt_processing
import pyabf as abf  # https://pypi.org/project/pyabf/
import pims_nd2
import pandas as pd
import datetime
import pytz  # for timezones
import numpy as np
import warnings
from two_photon_session import TwoPhotonSession
# TODO: open_dir opens dialog in foreground (in Jupyter), thanks to root.attributes("-topmost", True). Implement this in other dialog callign functions!


def raise_above_all(window):
    window.attributes('-topmost', 1)
    window.attributes('-topmost', 0)


def open_file(title: str = "Select file") -> str:
    """
    Opens a tkinter dialog to select a file. Returns the path of the file.
    :param title: The message to display in the open directory dialog.
    :return: the absolute path of the directory selected.
    """
    root = Tk()
    # dialog should open on top. Only works for Windows?
    root.attributes("-topmost", True)
    root.withdraw()  # keep root window from appearing
    return askopenfilename(title=title)


def open_dir(title: str = "Select data directory") -> str:
    """
    Opens a tkinter dialog to select a folder. Returns the path of the folder.
    :param title: The message to display in the open directory dialog.
    :return: the absolute path of the directory selected.
    """
    root = Tk()
    # dialog should open on top. Only works for Windows?
    root.attributes("-topmost", True)
    root.withdraw()  # keep root window from appearing
    folder_path = askdirectory(title=title)
    return folder_path


def open_session(data_path: str) -> TwoPhotonSession:
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
