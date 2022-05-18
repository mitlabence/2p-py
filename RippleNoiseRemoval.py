import pims_nd2
import numpy as np
from matplotlib import pyplot as plt
from nd2_to_caiman import np_arr_from_nd2
from pathos.multiprocessing import ProcessingPool as Pool

"""
# optional: show original figure of which plot is made
fig = plt.figure(figsize=(18,18))
plt.pcolormesh(nd2_data[0])
plt.show()
"""


def rnr_frame(frame, win, amplitude_threshold):  # endx and endy should be
    freq_image = np.fft.fftshift(np.fft.fft2(frame))  # make FFT
    # get log amplitude to detect spikes in fft
    ampl_image = np.log(np.abs(freq_image))
    end_x = ampl_image.shape[0]
    end_y = ampl_image.shape[1]
    bright_spikes = ampl_image > amplitude_threshold
    bright_spikes[round(end_x/2-win):round(end_x/2+win),
                  round(end_y/2-win):round(end_y/2+win)] = 0
    freq_image[bright_spikes] = 0
    filt_image = np.fft.ifft2(freq_image)
    return np.abs(filt_image)


def plot_rnr(frame, win, amplitude_threshold):  # ex. frame = nd2_data[0, :, :]
    freq_matrix = np.fft.fftshift(np.fft.fft2(frame))
    amplitude_image = np.log(np.abs(freq_matrix))
    bright_spikes = amplitude_image > amplitude_threshold  # default amplitude threshold
    rectangle_filter_boundary = np.zeros(amplitude_image.shape)
    filtered_spikes = np.copy(bright_spikes)
    end_y = amplitude_image.shape[0]  # todo: maybe switched?
    end_x = amplitude_image.shape[1]

    # plot rectangle that will be filtered out
    rectangle_filter_boundary[round(
        end_x/2 - win):round(end_x/2 + win), round(end_y/2 - win)] = 1
    rectangle_filter_boundary[round(
        end_x/2 - win):round(end_x/2 + win), round(end_y/2 + win)] = 1
    rectangle_filter_boundary[round(end_x/2 - win),
                              round(end_y/2 - win):round(end_y/2 + win)] = 1
    rectangle_filter_boundary[round(end_x/2 + win),
                              round(end_y/2 - win):round(end_y/2 + win)] = 1
    filtered_spikes[round(end_x/2-win):round(end_x/2+win),
                    round(end_y/2-win):round(end_y/2+win)] = 0
    # filter out whole fft spectrum, show raw and filtered
    freq_filtered = np.copy(freq_matrix)
    freq_filtered[bright_spikes] = 0.0

    filtered_image = np.abs(np.fft.ifft2(freq_filtered))

    # plot various data
    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    axs[0, 0].pcolormesh(amplitude_image)
    axs[0, 1].pcolormesh(bright_spikes + rectangle_filter_boundary)
    axs[0, 2].pcolormesh(np.abs(freq_filtered))
    axs[1, 0].pcolormesh(filtered_spikes)
    for i_row in range(len(amplitude_image)):
        axs[1, 1].plot(amplitude_image[i_row, :])
    axs[1, 1].axhline(y=amplitude_threshold, color='r', linestyle='-')
    # axs[0,0].colorbar()
    axs[1, 2].pcolormesh(np.log(np.abs(freq_matrix - freq_filtered)))
    # plot original and raw
    axs[2, 0].pcolormesh(frame)
    axs[2, 1].pcolormesh(filtered_image)
    plt.show()


def rnr():
    nd2_fpath = 'D:/PhD/Data/T386_MatlabTest/T386_20211202_green.nd2'
    win = 40
    amplitude_threshold = 10.8

    nd2_data = np_arr_from_nd2(nd2_fpath)

    filtered_data = np.zeros(nd2_data.shape)
    for i_frame in range(nd2_data.shape[0]):
        filtered_data[i_frame] = rnr_frame(
            nd2_data[i_frame, :, :], win, amplitude_threshold)
    return filtered_data



	
	