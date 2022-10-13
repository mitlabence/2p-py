from tkinter import Tk  # use tkinter to open files
from tkinter import simpledialog


def text_input_dialog(window_title: str = "Prompt window", prompt="Enter text") -> str:
    """
    Pop-up window with text input, used e.g. for acquiring metadata.
    :return: the user-specified text as a string
    """
    root = Tk()
    root.geometry("1024x768")
    root.withdraw()
    root.update_idletasks()
    return simpledialog.askstring(window_title, prompt + "\t\t\t\t\t\t", parent=root)  # add tabs to increase window size