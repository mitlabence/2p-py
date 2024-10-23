import sys
import os
try:
    # Get the absolute path of the parent directory (the root folder)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Add the libs folder to the system path
    sys.path.insert(0, root_dir)
finally:
    import labrotation.file_handling as fh
    import labrotation.two_photon_session as tps

# TODO: make a very small dataset: LFP, labview and nd2 with both only green and green + red.
# TODO: make it a proper test file in the future (pytest). Need small data first.

FOLDER = os.path.normpath("D:\PhD\Data\T386_MatlabTest")

ND2_GREEN_FNAME = "T386_20211202_green.nd2"
ND2_GREEN_LFP = "21d02000.abf"
ND2_GREEN_LV = "T386.021221.1105.txt"
ND2_GREEN_LVTIME = "T386.021221.1105time.txt"
ND2_GREEN_NIK = "T386.021221.1105_nik.txt"

ND2_DUAL_FNAME = "T386_20211202_green_red.nd2"
ND2_DUAL_LFP = "21d02001.abf"
ND2_DUAL_LV = "T386.021221.1106.txt"
ND2_DUAL_LVTIME = "T386.021221.1106time.txt"
ND2_DUAL_NIK = "T386.021221.1106_nik.txt"

# TODO: fix this somehow to not ask for folder but also not push changed value each time pushing from a different PC
MATLAB_2P_PATH = os.path.normpath("D:\\PhD\\matlab-2p\\matlab-2p\\")

# only green channel
nd2_green_fpath = os.path.join(FOLDER, ND2_GREEN_FNAME)
nd2_green_lfp_fpath = os.path.join(FOLDER, ND2_GREEN_LFP)
nd2_green_lv_fpath = os.path.join(FOLDER, ND2_GREEN_LV)
nd2_green_lvtime_fpath = os.path.join(FOLDER, ND2_GREEN_LVTIME)
nd2_green_nik_fpath = os.path.join(FOLDER, ND2_GREEN_NIK)

# export json files (for testing reopening)
nd2_green_full_json = os.path.join(FOLDER, "GREEN_full.json")
nd2_green_nolfp_json = os.path.join(FOLDER, "GREEN_nolfp.json")

# green and red channels
nd2_dual_fpath = os.path.join(FOLDER, ND2_DUAL_FNAME)
nd2_dual_lfp_fpath = os.path.join(FOLDER, ND2_DUAL_LFP)
nd2_dual_lv_fpath = os.path.join(FOLDER, ND2_DUAL_LV)
nd2_dual_lvtime_fpath = os.path.join(FOLDER, ND2_DUAL_LVTIME)
nd2_dual_nik_fpath = os.path.join(FOLDER, ND2_DUAL_NIK)

nd2_dual_full_json = os.path.join(FOLDER, "DUAL_full.json")
nd2_dual_nolfp_json = os.path.join(FOLDER, "DUAL_nolfp.json")


def test_tps_open_files():
    # test with all files available, only green
    session = tps.TwoPhotonSession.init_and_process(nd2_path=nd2_green_fpath,
                                                    nd2_timestamps_path=nd2_green_nik_fpath,
                                                    labview_path=nd2_green_lv_fpath,
                                                    labview_timestamps_path=nd2_green_lvtime_fpath,
                                                    lfp_path=nd2_green_lfp_fpath,
                                                    matlab_2p_folder=MATLAB_2P_PATH)
    # session.export_json(fpath=nd2_green_full_json)
    # TODO: test class fields (for all cases), their shape, type ...
    print(vars(session).keys())
    print("Testing TwoPhotonSession with all parameters done.")
    # test without LFP
    session = tps.TwoPhotonSession.init_and_process(nd2_path=nd2_green_fpath,
                                                    nd2_timestamps_path=nd2_green_nik_fpath,
                                                    labview_path=nd2_green_lv_fpath,
                                                    labview_timestamps_path=nd2_green_lvtime_fpath,
                                                    matlab_2p_folder=MATLAB_2P_PATH)
    # session.export_json(fpath=nd2_green_nolfp_json)
    print("Testing TwoPhotonSession with no LFP done.")
    # print(vars(session).keys())

    session = tps.TwoPhotonSession.init_and_process(nd2_path=nd2_dual_fpath,
                                                    nd2_timestamps_path=nd2_dual_nik_fpath,
                                                    labview_path=nd2_dual_lv_fpath,
                                                    labview_timestamps_path=nd2_dual_lvtime_fpath,
                                                    lfp_path=nd2_dual_lfp_fpath,
                                                    matlab_2p_folder=MATLAB_2P_PATH)
    # session.export_json(fpath=nd2_dual_full_json)
    print("Testing TwoPhotonSession with all parameters (dual channel) done.")

    session = tps.TwoPhotonSession.init_and_process(nd2_path=nd2_dual_fpath,
                                                    nd2_timestamps_path=nd2_dual_nik_fpath,
                                                    labview_path=nd2_dual_lv_fpath,
                                                    labview_timestamps_path=nd2_dual_lvtime_fpath,
                                                    matlab_2p_folder=MATLAB_2P_PATH)
    # session.export_json(fpath=nd2_dual_nolfp_json)
    print("Testing TwoPhotonSession with all parameters (dual channel) done.")

    # test importing sessions
    # session = tps.TwoPhotonSession.from_json(nd2_green_full_json)
    # TODO: test if imported session has proper functionality
    # session = tps.TwoPhotonSession.from_json(nd2_green_nolfp_json)

    # session = tps.TwoPhotonSession.from_json(nd2_dual_full_json)

    # session = tps.TwoPhotonSession.from_json(nd2_dual_full_json)


# test_tps_open_files()
