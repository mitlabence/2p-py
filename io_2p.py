from tkinter import Tk  # use tkinter to open files
from tkinter import simpledialog


def text_input_dialog(window_title: str = "Prompt window", prompt="Enter text") -> str:
    """
    Pop-up window with text input, used e.g. for acquiring metadata.
    :return: the user-specified text as a string
    """
    text = ""
    root = Tk()
    root.geometry("1024x768")
    # FIXME: askstring does not take over geometry of parent... post an issue on tkinter github?
    #       Or just overwrite askstring and parent classes to allow for geometry parameter
    root.withdraw()
    root.update_idletasks()
    simpledialog.askstring(window_title, prompt, parent=root)
    return text
