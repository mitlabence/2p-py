import sys
import os
# Get the absolute path of the parent directory (the root folder)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the libs folder to the system path
sys.path.insert(0, root_dir)


def test_imports():
    import labrotation.file_handling as fh
    import labrotation.two_photon_session as tps
