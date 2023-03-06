import matlab.engine
from labrotation.file_handling import open_dir, open_file
import os.path

# The structure of a labview output file, decoded from the labview file (Movementdetection.vi, Integrator.vi) by column:
# 1.  Rounds
# 2.  Speed
# 3.  Total distance
# 4.  DistancePR (per round)
# 5.  Reflectivity
# 6.  Lick detection
# 7.  Stripes in total
# 8.  Stripes in round
# 9.  Total time
# 10. Time per round
# 11. Stimuli
# ... Stimuli
# 19. Stimuli
# 20. Pupil area

def beltProcessPipeline(belt_path: str, matlab_2p_folder: str, nargout: int = 1) -> dict:
    # TODO: add nikon_ts_path here as well, see beltProcessPipelineExpProps
    eng = matlab.engine.start_matlab()
    # dialog window pops up in background!
    if matlab_2p_folder is None:
        matlab_2p_folder = open_dir("Open matlab-2p folder")
    m2p_path = eng.genpath(matlab_2p_folder)
    eng.addpath(m2p_path, nargout=0)
    if belt_path is None:
        # result is a dictionary.
        return eng.beltProcessPipeline(nargout=nargout)
    else:
        if not (belt_path[-4:] == ".txt"):
            raise Exception("Error: belt_path is not a .txt file.")
        belt_path, belt_fname = os.path.split(belt_path)
        # TODO: need to check if these assumed files exist before passing them to matlab!
        nikon_fname = belt_fname + "_nik"
        return eng.beltProcessPipeline(belt_path, belt_fname, nikon_fname, nargout=nargout)


def beltProcessPipelineExpProps(belt_path: str, nikon_ts_path: str, matlab_2p_folder: str, nargout: int = 3) -> dict:
    eng = matlab.engine.start_matlab()
    # dialog window pops up in background!
    if matlab_2p_folder is None:
        matlab_2p_folder = open_dir("Open matlab-2p folder")
    m2p_path = eng.genpath(matlab_2p_folder)
    eng.addpath(m2p_path, nargout=0)
    if belt_path is None:
        # result is a dictionary.
        print("Calling Matlab-2p beltProcessPipelineExpProps without arguments")
        return eng.beltProcessPipelineExpProps(nargout=nargout)
    else:
        if not (os.path.splitext(belt_path)[-1] == ".txt"):
            raise Exception("beltProcessPipelineExpProps - Error: belt_path is not a .txt file.")
        belt_path, belt_fname = os.path.split(belt_path)
        # get rid of extension
        belt_fname = os.path.splitext(belt_fname)[0]
        # TODO: need to check if these assumed files exist before passing them to matlab!
        if not (os.path.exists(nikon_ts_path)):
            nikon_ts_path = open_file(
                f"beltProcessPipelineExpProps - Nikon metadata {nikon_ts_path} not found. Please open it now.")
        nikon_fname = os.path.splitext(os.path.split(nikon_ts_path)[-1])[0]
        # TODO: belt and nikon must be in one folder, this looks limiting...
        # TODO: why actually get rid of extension?
        # TODO: I think the problem is the "/" at the end is missing in belt_path. Test in Matlab.
        print(
            f"Calling Matlab-2p beltProcessPipelineExpProps with arguments:\n\t{belt_path}\n\t{belt_fname}\n\t"
            f"{nikon_fname}")
        return eng.beltProcessPipelineExpProps(belt_path, belt_fname, nikon_fname, nargout=nargout)
