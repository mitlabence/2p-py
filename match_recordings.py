import tkinter as tk
import pims_nd2


def compare_recordings(nd2_1_path, nd2_2_path, N_frames):
    nd2_1 = pims_nd2.ND2_Reader(nd2_1_path)
    nd2_2 = pims_nd2.ND2_Reader(nd2_2_path)
    # create average frames
    frame_1 = nd2_1[0]
    frame_2 = nd2_2[0]
    for i in range(1, N_frames):
        frame_1 += nd2_1[i]
        frame_2 += nd2_2[i]
    frame_1 = frame_1/N_frames
    frame_2 = frame_2/N_frames
