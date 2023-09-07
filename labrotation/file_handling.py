from tkinter import Tk  # use tkinter to open files
from tkinter.filedialog import askopenfilename, askdirectory
import os.path
import datetime as dt


# TODO: open_dir opens dialog in foreground (in Jupyter), thanks to root.attributes("-topmost", True). Implement this
#  in other dialog callign functions!


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
    return os.path.normpath(askopenfilename(title=title))


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
    return os.path.normpath(folder_path)


def choose_dir_for_saving_file(title: str = "Select a folder to save the file to", fname: str = "output_file.txt"):
    """
    Opens a tkinter dialog to select a folder. Returns opened folder + file name as path string.
    :param title:
    :param fname:
    :return:
    """
    return os.path.normpath(os.path.join(open_dir(title), fname))


def get_filename_with_date(raw_filename: str = "output_file", extension: str = ".txt"):
    """
    Given a root filename raw_filename, create an extended filename with extension. This avoids overwriting files saved
    repeatedly to the same folder by appending the date and time (including the seconds).
    :param raw_filename: file name without extension
    :param extension: the desired file extension. It should include the '.'!
    :return:
    """
    # todo: this should be a bit more sophisticated. (dealing with cases like extension without "." etc.), getting rid
    #  of extension in raw_filename if supplied...
    datetime_suffix = get_datetime_for_fname()  # dt.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    return raw_filename + "_" + datetime_suffix + extension


def get_datetime_for_fname():
    now = dt.datetime.now()
    return f"{now.year:04d}{now.month:02d}{now.day:02d}-{now.hour:02d}{now.minute:02d}{now.second:02d}"