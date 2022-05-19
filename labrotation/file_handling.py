from tkinter import Tk  # use tkinter to open files
from tkinter.filedialog import askopenfilename, askdirectory
import os.path
import pyabf as abf  # https://pypi.org/project/pyabf/
import pims_nd2
import pandas as pd
import datetime
import pytz  # for timezones
import numpy as np
import warnings
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


def open_dir(title: str = "Select data directory", ending_slash: bool = False) -> str:
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
    if ending_slash:
        folder_path += "/"
    return folder_path
