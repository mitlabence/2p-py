2p-py
Installation:
1. Install anaconda.
2. Create an environment (mamba is difficult or impossible to install without an environment) with arbitrary name. I will call it xy. Then use conda activate xy to activate it.
3. Install mamba in the environment with conda install -c conda-forge mamba
4. Install CaImAN. https://github.com/flatironinstitute/CaImAn This will create an environment inside xy with mamba: 
	mamba create -n caiman -c conda-forge caiman
5. conda activate caiman when in the environment xy.
6. Check python version by starting python (command: "python"). The version is most important for the Matlab Engine (running matlab code in python), see https://de.mathworks.com/help/matlab/matlab-engine-for-python.html Versions of python compatible with matlab by release.
7. Install opencv (cv2) python package when inside the xy or caiman environment: pip install opencv-python (Windows; dirty solution, but conda installation is difficult as always)
8. Enter 2p-py directory and make sure caiman environment is active.
9. Downgrade jinja2 to 3.0.3 (2022.03.28: 3.1.1 is the current version). The reason is that "from jinja2 import Markup" had been deprecated for a long time, 
	and was removed recently. Now the solution would be to import from markupsafe directly, which is not implemented in caiman, and the environment does not fix the jinja2 version...
10. while caiman env is active: pip install pims_nd2
11. jupyter-notebook