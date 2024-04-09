from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from math import ceil, floor, pi
import cmath
import numpy as np
# some functions are just weird in matlab and cannot be easily reproduced in python, such as histcounts().
import matlab

# Given the hdf5 file paths of the CaImAn results and the corresponding exported TwoPhotonSession object, this function performs the place coding analysis.
eng = matlab.engine.start_matlab()

# TODO: change caim input to actually needed fields of caim!!!
# TODO: caim["S_bin"] etc. might have to be replaced with caim["S_bin"].copy()! Otherwise original fields will be overwritten (when list)!!!


def open_caim(caim_hdf5_path) -> dict:  # TODO: implement proper opening
    caim_data = loadmat("D:\\Downloads\\caim.mat")
    caim_data = caim_data["caim"]
    caim_C = caim_data["C"][0][0]
    # TODO: Df in caiman-MATLAB does not exist in caiman python...
    caim_Df = caim_data["Df"][0][0]
    caim_S = caim_data["S"][0][0]
    caim = {"C": caim_C, "Df": caim_Df, "S": caim_S}
    return caim


def open_scn(two_photon_session_hdf5_fpath) -> dict:  # TODO: implement proper opening
    scn_data = loadmat("D:\\Downloads\\scn.mat")
    scn_data = scn_data["scn"]
    scn_rounds = np.ravel(scn_data["rounds"][0][0])
    scn_distance = np.ravel(scn_data["distance"][0][0])
    scn_running = np.ravel(scn_data["running"][0][0])
    scn_speed = np.ravel(scn_data["speed"][0][0])
    scn = {"speed": scn_speed, "rounds": scn_rounds,
           "distance": scn_distance, "running": scn_running}
    return scn


def apply_bins(scn):
    """
    Parameters
    ----------
    scn : _type_
        _description_
    """
    n_rounds = np.max(scn["rounds"])

    print(f"The mouse ran {n_rounds} rounds.")

    bin_size_mm = 10.  # size of bin in mm
    n_bins = ceil(np.max(scn["distance"])/bin_size_mm)
    starting_round = 0

    binning_shape = (np.max(scn["rounds"]), n_bins)

    speed_bin = np.zeros(binning_shape)
    wait_bin = np.zeros(binning_shape)

    for i_round in range(starting_round, n_rounds):
        running_current_round = scn["running"][scn["rounds"] == i_round] == 1
        distance_current_round = scn["distance"][scn["rounds"] == i_round]
        speed_current_round = scn["speed"][scn["rounds"] == i_round]
        for i_bin in range(n_bins):
            # FIXME: why distance_current_round > i_bin*bin_size_mm + 1? Why the +1?
            current_bin_indices = np.logical_and(
                distance_current_round > i_bin*bin_size_mm + 1, distance_current_round < (i_bin+1)*bin_size_mm)

            speed_current_bin = speed_current_round[current_bin_indices]
            # TODO: deal with negative values too? That also shows locomotion...
            speed_current_bin = speed_current_bin[speed_current_bin > 0.]
            if len(speed_current_bin) > 0:
                speed_bin[i_round, i_bin] = np.mean(speed_current_bin)
            if np.sum(current_bin_indices) > 0:
                wait_bin[i_round, i_bin] = np.sum(
                    running_current_round[current_bin_indices] == 0)
    return (speed_bin, wait_bin)


# Speed coding
def speed_corr(caim, scn):
    speed = scn["speed"].copy()*100  # conversion to cm/s?
    speed[speed < -20] = 0  # discard backwards locomotion
    speed = np.concat([speed, np.array(speed[-1])])
    # TODO: originally, 30 window size was set... what happens then? (window size needs to be odd for moving average)
    speed = smooth_matlab_like(speed, 31)
    # cut to number of caiman frames; should not change anything...
    speed = speed[:caim["C"].shape[1]]
    # TODO: why filter twice? First -20, then smooth, then filter below 0...
    speed[speed <= 0.] = 0.

    # TODO: what if min(speed) is not 0? Then this results in more than 20 (21 incl. max(speed)) bins
    bin_limits = np.arange(np.min(speed), np.max(speed)+0.1, np.max(speed)/20.)
    n_bins = len(bin_limits)-1
    n_cells = caim["C"].shape[0]

    # the mean of temporal trace where speed falls within given bin, divided by Df
    traces = np.zeros((n_cells, n_bins))
    onsets = np.zeros((n_cells, n_bins))
    trace_errors = np.zeros(n_cells, n_bins)
    for i_bin in range(n_bins - 1):
        for i_neuron in range(n_cells):
            # get the speed limits corresponding to the bin
            lower_speed = bin_limits[i_bin]
            upper_speed = bin_limits[i_bin + 1]
            bin_indices = np.logical_and(
                speed >= lower_speed, speed < upper_speed)
            # FIXME: Df does not exist in caiman python data!
            traces[i_neuron, i_bin] = np.mean(
                caim["C"][i_neuron][bin_indices] / caim["Df"][i_neuron])
            trace_errors[i_neuron, i_bin] = np.std(
                caim["C"][i_neuron][bin_indices] / caim["Df"][i_neuron])
            onsets[i_neuron, i_bin] = np.sum(caim["S_bin"])
    assert len(traces.shape) > 1  # traces 2D array
    corr_coeffs = []
    p_vals = []
    for neuron_trace in traces:
        c, p = pearsonr(bin_limits[1:], neuron_trace)
        corr_coeffs.append(c)
        p_vals.append(p)
    # TODO compare form of corr_coeffs and p_vals with c and p after/before reshape in placecor.m, lines 596-598!
    return (traces, trace_errors, onsets, corr_coeffs, p_vals, bin_limits)


def smooth_matlab_like(arr: np.array, win_size: int = 5) -> np.array:
    """Smooth an array as the matlab function smooth() does. See https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python

    Parameters
    ----------
    arr : np.array
       NumPy 1-D array containing the data to be smoothed
    win_size : int
        smoothing window size needs, which must be odd number, as in the original MATLAB implementation
    Returns
    -------
    np.array
        The smoothed array
    """
    out0 = np.convolve(arr, np.ones(win_size, dtype=int), 'valid')/win_size
    r = np.arange(1, win_size-1, 2)
    start = np.cumsum(arr[:win_size-1])[::2]/r
    stop = (np.cumsum(arr[:-win_size:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def gauss(x, *p):
    a, b, c = p  # https://en.wikipedia.org/wiki/Gaussian_function
    return a*np.exp(-0.5*(x-b)**2/c**2)


def normalize_caim(caim):
    threshold = np.zeros(caim["C"].shape)
    sigma = 6
    n_bins = 50
    # Decrease window size for descrete adaptation of threshold (smooth threshold scroll down)
    window_size = 6000
    n_windows = ceil(caim["C"].shape[1]/window_size)
    counts = np.zeros((n_windows, n_bins))
    centers = np.zeros((n_windows, n_bins))
    threshold_temp = np.zeros(n_windows)

    for i_neuron in range(caim["S"].shape[0]):
        for i_window in range(n_windows+1):
            if i_window < n_windows:
                current_window = range(
                    i_window*window_size, (i_window+1)*window_size)
            else:
                current_window = range(
                    i_window*window_size, caim["S"].shape[1])
            S_windowed = caim["S"][i_neuron][current_window].copy()
            # TODO: make sure the type is correct!
            S_windowed_matlab = matlab.double(S_windowed.tolist())
            (N, edges) = eng.histcounts(S_windowed_matlab, n_bins, nargout=2)
            counts[i_window, :] = N
            # FIXME: this is obviously not correct... they are edges, not centers
            centers[i_window, :] = edges[:-1]

        for i_window in range(n_windows):
            x = centers[i_window, :]
            y = counts[i_window, :]
            # TODO: make sure this always returns only one value!
            i_max_count = np.argmax(y)
            # TODO: idea behind starting parameters?
            p0 = [y[i_max_count], x[i_max_count], 10]
            coeffs, var_matrix = curve_fit(
                gauss, x, y, bounds=([0, 0, 0], [1000, 0, 1000]), p0=p0)
            # threshtemp(j) = f1.b+sigmamult*abs(f1.c);
            threshold_temp[i_window] = coeffs[1] + \
                sigma*abs(coeffs[2])  # b + sigma*abs(c)

        for i_window in range(n_windows+1):
            if i_window < n_windows:
                current_window = range(
                    i_window*window_size, (i_window+1)*window_size)
            else:
                current_window = range(
                    i_window*window_size, caim["S"].shape[1])
            threshold[i_neuron, current_window] = threshold_temp[i_window]

    S_norm = np.zeros(caim["S"].shape)
    S_bin = np.zeros(caim["S"].shape)

    for i_neuron in range(caim["S"].shape[0]):  # TODO: n_neurons...
        # FIXME: probably not working. Decode what "list > [another list]" means
        S_bin[i_neuron, caim["S"][i_neuron] > threshold[i_neuron, :]] = 1
        S_norm[i_neuron] = caim["S"][i_neuron]*S_bin[i_neuron]
        S_norm[i_neuron] = S_norm[i_neuron]/np.max(S_norm[i_neuron])

    S_sparse = S_bin.copy()
    for i_neuron in range(S_bin.shape[0]):  # TODO: n_neurons...
        nonzero_indices = np.nonzero(S_bin[i_neuron])[0]
        # TODO: decode this mess... "sparsen inter spike interval"
        S_sparse[i_neuron, nonzero_indices[np.concatenate(
            [np.diff(nonzero_indices) == 1, np.array([False])])] + 1] = 0
    S_bin = S_sparse
    S_norm = S_bin*S_norm
    return (S_bin, S_norm, threshold)


def ca_on_place(caim, scn, spatial_binsize: int, min_ev, cell_ID=None):
    n_bins = ceil(np.max(scn["distance"]/spatial_binsize))
    n_neurons = caim["C"].shape[0]
    # 0 until max entry in rounds = max + 1 rounds
    n_rounds = np.max(scn["rounds"]) + 1
    spatial_code = np.zeros((n_neurons, n_bins, n_rounds))
    starting_round = 0  # double(scn.distance(1)>scn.distance(end));
    for i_neuron in range(n_neurons):  # for j = 1 : size(caim.S_bin,1)
        for current_round in range(starting_round, n_rounds):
            distance_current_round = scn["distance"][np.logical_and(
                scn["running"] == 1, scn["rounds"] == current_round)].copy()
            S_bin_current_round = caim["S_bin"][i_neuron][np.logical_and(
                scn["running"] == 1, scn["rounds"] == current_round)].copy()
        # FIXME: what is the purpose of this?
        if current_round < np.max(scn["rounds"]):
            for i_bin in range(n_bins):
                # TODO: what is the +1 here? why no >= ?
                bin_range = np.where(np.logical_and(distance_current_round > i_bin *
                                     spatial_binsize+1, distance_current_round < (i_bin+1)*spatial_binsize))
                if len(bin_range) > 0:
                    spatial_code[i_neuron, i_bin, current_round] = np.sum(
                        S_bin_current_round[bin_range])
        elif starting_round == 0:  # FIXME: what is this weird elif?
            for i_bin in range(floor(scn["distance"][0]/spatial_binsize)):
                bin_range = np.where(np.logical_and(distance_current_round > i_bin *
                                     spatial_binsize+1, distance_current_round < (i_bin+1)*spatial_binsize))
                if len(bin_range) > 0:
                    spatial_code[i_neuron, i_bin, current_round] = np.mean(
                        S_bin_current_round[bin_range])  # FIXME: why mean here and sum above?
    if cell_ID is None:
        # FIXME: why binarize only if cell_ID is not specified?
        spatial_code[spatial_code > 1] = 1
        spatial_copy = spatial_code.copy()
        # axis=2 in matlab. keepdims=True to keep number of axes (summed axis has 1 length now)
        spatial_copy = np.sum(spatial_copy, axis=1, keepdims=True)
        spatial_copy[spatial_copy > 1] = 1
        # FIXME: the 3rd axis should disappear on the first sum, leading to an error here... should be fixed with keepdims=True now
        # TODO: keepdims=True here? check output cell_ID format... is it good?
        spatial_copy = np.sum(spatial_copy, 2)
        cell_ID = np.where(spatial_copy > min_ev)
    spatial_code_sorted = spatial_code[cell_ID, :, :]
    return (spatial_code, spatial_code_sorted, n_bins, cell_ID)


def shuffle_data(caim):
    S_bin = caim["S_bin"].copy()
    # Shuffle entries for individual cells (?)
    for i_cell in range(len(S_bin)):
        S_bin[i_cell] = np.random.permutation(S_bin[i_cell])
    # Randomly introduce shift in neurons (?)
    C = np.zeros(caim["C"].shape, caim["C"].dtype)
    for i_cell in range(len(S_bin)):
        # select a random point the beginning should be shifted to
        i_shifted_begin = np.random.randint(0, len(S_bin[i_cell]))
        len_first_segment = len(caim["C"][i_cell][i_shifted_begin:])
        C[i_cell][:len_first_segment] = caim["C"][i_cell][i_shifted_begin:]
        C[i_cell][len_first_segment:] = caim["C"][i_cell][:i_shifted_begin]
    return (S_bin, C)


# FIXME: rename rnd and o once it is understood wtf they stand for
def losonczy(spatial_code_sorted, n_bins, rnd, o):
    space_coding_polar = np.zeros((spatial_code_sorted.shape[0], 3))
    alpha = np.linspace(0, 2*pi, n_bins)
    theta = np.linspace(0, 2*(len(rnd)+1)*pi, n_bins*(len(rnd)+1))
    rho = np.arange(1, len(theta)+1)*(len(rnd)+1)/len(theta)

    n_cells = len(spatial_code_sorted)
    n_rounds = spatial_code_sorted.shape[2]

    for i_cell in range(n_cells):
        c = []
        d = []
        dd = []  # TODO: give names to these once their role is understood
        for current_round in range(n_rounds):
            # spatial_code_sorted has dimensions (n_neurons, n_bins, n_rounds)
            # get all bins for current cell and current round
            # TODO: test if indexing here is correct
            spatial_code_bins = np.where(
                spatial_code_sorted[i_cell, :, current_round])
            for bin in spatial_code_bins:
                # TODO: 1-indexing or 0-indexing current_round here? if no more work with it, then 1-indexing is what matlab is using, for consistency could switch to that...
                c.append([alpha[bin], rho[n_bins*current_round + bin],
                         current_round, spatial_code_sorted[i_cell, bin, current_round]])
                d.append([cmath.exp(alpha[bin]*1j)/o[bin]])
                dd.append(o[bin])
        # ? what does this look like?
        place_coding_vector = np.sum(dd)*np.sum(d)/len(d)**2
        # alpha, rho, round, count in bin
        space_coding_polar.append(
            [place_coding_vector, len(place_coding_vector), c])
    # TODO: hard-coded numbers... 150 appeared somewhere already. 1500 is total belt length
    length_place_field = round(150/1500*spatial_code_sorted.shape[1])
    # TODO: check shape here! spcsorttemp in matlab seems to have one duplicate row
    # extend spatial code data to deal with boundaries
    spatial_code_sorted_extended = np.concatenate(
        [spatial_code_sorted[:, -length_place_field-1:, :], spatial_code_sorted, spatial_code_sorted[:, :length_place_field, :]], axis=1)
    first_round = np.zeros(n_cells)
    for i_cell in range(n_cells):  # TODO: understand algorithm, write comments
        min_angle = np.where(abs(alpha - cmath.phase(space_coding_polar[i_cell, 0])) == np.min(
            abs(alpha - cmath.phase(space_coding_polar[i_cell, 0]))))
        permuted = np.transpose(
            np.sum(spatial_code_sorted_extended[i_cell, min_angle:min_angle+length_place_field, :], axis=1), [2, 1, 0])  # FIXME: np.sum() reduces number of axes!
        # there is at least 1 non-zero element
        if len(np.nonzero(permuted)[0]) > 0:
            first_round[i_cell] = np.nonzero(permuted)[0][0]
    return (space_coding_polar, theta, rho, first_round)


def dombeck(spatial_code_sorted, length_place_field):
    n_cells = spatial_code_sorted.shape[0]
    n_bins = spatial_code_sorted.shape[1]
    n_rounds = spatial_code_sorted.shape[2]
    # extend spatial coding to deal with boundaries
    spatial_code_sorted_extended = np.concatenate(
        [spatial_code_sorted[:, -length_place_field-1:, :], spatial_code_sorted, spatial_code_sorted[:, :length_place_field, :]], axis=1)
    # why 5 the last dim? number of criteria in algorithm?
    place_field = np.zeros((n_cells, n_bins, 5))
    center_field = np.zeros(n_cells)

    # Criterion 1: activity in at least percent_rounds% of rounds
    # percentage of rounds where cell shows place activity. TODO: is it percent (0.2%) or ratio (20%)? I bet latter
    percent_rounds = 0.2
    for i_bin in range(n_bins):
        # TODO: in numpy, sum() removes the axis along which summation was formed. keepdims=True should induce matlab-like behaviour (at least for higher dimensions)
        place_field[:, i_bin, 0] = np.mean(np.sum(
            spatial_code_sorted_extended[:, i_bin:i_bin+2*length_place_field+1, :], axis=1, keepdims=True), axis=2)
        temp = np.sum(
            spatial_code_sorted_extended[:, i_bin:i_bin+2*length_place_field+1, :], 1, keepdims=True)
        temp[temp > 1] = 1
        temp = np.sum(temp, 2, keepdims=True)/n_rounds
        temp[temp < percent_rounds] = 0
        temp[temp > 0] = 1
        # TODO: play with the dimensions if necessary, until result is same or matlab, or makes sense
        place_field[:, i_bin, 1] = temp
    # Criterion 2: activity at least [activity_threshold] times higher within place field
    activity_threshold = 7
    for i_bin in range(n_bins):
        # TODO: maybe off-by-one here...
        place_field[:, i_bin, 2] = np.mean(np.concatenate(
            [place_field[:, :i_bin-length_place_field+1, 0], place_field[:, i_bin+length_place_field:, 0]], axis=1))
    temp = activity_threshold * place_field[:, :, 2] < place_field[:, :, 0]
    place_field[:, :, 2] = temp

    # Combine criteria 1 and 2
    place_field[:, :, 3] = place_field[:, :, 0] * \
        place_field[:, :, 1] * place_field[:, :, 2]

    # Criterion 3: place fields need to have minimal length
    n_fields = temp.shape[1]
    for i_field in range(n_fields):
        


scn = open_scn("")
caim = open_caim("")


speed_bin, wait_bin = apply_bins(scn)


# Correlate activity with Ca response in bin on belt
min_rounds = 5
min_ev = 3

# TODO: assert "C" in caim_data!

if np.max(n_rounds) < min_rounds:
    # TODO: implement returning empty fields as alternative here? See placecor.m lines 68-76
    raise Exception("Not enough rounds!")
