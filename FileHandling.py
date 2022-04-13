import os
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import pims_nd2

def open_file(instruction: str = "Open file") -> str:
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(title=instruction)  # show an "Open" dialog box and return the path to the selected file
    return filename


def read_nd2(absolute_path):
    nd2 = pims_nd2.ND2_Reader(absolute_path)

