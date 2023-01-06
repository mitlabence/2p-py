import numpy as np
from matplotlib import pyplot as plt
from nd2_to_caiman import np_arr_from_nd2
from typing import Tuple
"""
# optional: show original figure of which plot is made
fig = plt.figure(figsize=(18,18))
plt.pcolormesh(nd2_data[0])
plt.show()
"""
# TODO: write assert() tests!
# TODO: write test on test data to compare parallel and single thread RNR,
#       also compare with Matlab RNR!
# TODO: make it work for 2-channel recordings too!
# TODO: make sure that encountering division by 0 does not affect end result (I guess then entry becomes INF, filtered out by threshold comparison! But check this!)


class RNR():
    def __init__(self, win, amplitude_threshold, n_threads: int = 1):
        self.win = win
        self.amplitude_threshold = amplitude_threshold
        self.end_x = 0
        self.end_y = 0
        self.n_frames = 0
        self.nd2_data = None  # raw data
        self.rnr_data = None  # contains RNR-processed data
        # TODO: if n_threads = 1, only do rnr_singlethread. Otherwise make sure computer has enough threads, perform parallel
        self.n_threads = n_threads
        # assert(self.n_threads <= cpu_count())

    def open_recording(self, nd2_fpath, begin_end_frames: Tuple[int, int]=None):
        self.nd2_data = np_arr_from_nd2(nd2_fpath, begin_end_frames)
        self.rnr_data = np.empty(self.nd2_data.shape, dtype=np.float64)
        self.n_frames = self.nd2_data.shape[0]
        self.end_x = self.nd2_data.shape[1]
        self.end_y = self.nd2_data.shape[2]
        print(
            f"Opened recording {self.end_x}x{self.end_y}, {self.n_frames} frames. Initialized empty results array.")

    def rnr_frame(self, frame):  # endx and endy should be
        freq_image = np.fft.fftshift(np.fft.fft2(
            frame))  # make FFT
        # get log amplitude to detect spikes in fft
        # D:\Codebase\2p-py\RippleNoiseRemoval.py:46: RuntimeWarning: divide by zero encountered in log ampl_image = np.log(np.abs(freq_image))
        ampl_image = np.log(np.abs(freq_image))
        bright_spikes = ampl_image > self.amplitude_threshold
        bright_spikes[round(self.end_x/2-self.win):round(self.end_x/2+self.win),
                      round(self.end_y/2-self.win):round(self.end_y/2+self.win)] = 0
        freq_image[bright_spikes] = 0
        filt_image = np.fft.ifft2(freq_image)
        return np.abs(filt_image)

    # ex. frame = nd2_data[0, :, :]
    def plot_rnr(self, i_frame: int = 0):
        assert(self.nd2_data is not None)
        freq_matrix = np.fft.fftshift(np.fft.fft2(self.nd2_data[i_frame]))
        amplitude_image = np.log(np.abs(freq_matrix))
        # default amplitude threshold
        bright_spikes = amplitude_image > self.amplitude_threshold
        rectangle_filter_boundary = np.zeros(amplitude_image.shape)
        filtered_spikes = np.copy(bright_spikes)
        end_y = amplitude_image.shape[0]  # todo: maybe switched?
        end_x = amplitude_image.shape[1]

        # plot rectangle that will be filtered out
        rectangle_filter_boundary[round(
            end_x/2 - win):round(self.end_x/2 + self.win), round(self.end_y/2 - self.win)] = 1
        rectangle_filter_boundary[round(
            end_x/2 - win):round(self.end_x/2 + self.win), round(self.end_y/2 + self.win)] = 1
        rectangle_filter_boundary[round(self.end_x/2 - self.win),
                                  round(self.end_y/2 - self.win):round(self.end_y/2 + self.win)] = 1
        rectangle_filter_boundary[round(self.end_x/2 + self.win),
                                  round(self.end_y/2 - self.win):round(self.end_y/2 + self.win)] = 1
        filtered_spikes[round(self.end_x/2-self.win):round(self.end_x/2+self.win),
                        round(self.end_y/2-self.win):round(self.end_y/2+self.win)] = 0
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
        axs[1, 1].axhline(y=self.amplitude_threshold, color='r', linestyle='-')
        # axs[0,0].colorbar()
        axs[1, 2].pcolormesh(np.log(np.abs(freq_matrix - freq_filtered)))
        # plot original and raw
        axs[2, 0].pcolormesh(self.nd2_data[i_frame])
        axs[2, 1].pcolormesh(filtered_image)
        plt.show()

    def rnr_singlethread(self):
        for i_frame in range(self.n_frames):
            self.rnr_data[i_frame] = self.rnr_frame(
                self.nd2_data[i_frame, :, :])
        print("RNR completed.")
        return self.rnr_data

#    def rnr_parallel(self):
#        # TODO: there are some issues with starting the pool again: https://github.com/uqfoundation/pathos/issues/111
#        # FIXME: Running this method again results in ValueError: Pool not running. Possible solution: try ... except ValueError: pool.restart()
#        # TODO: parallel seems much slower!
#        p = Pool(self.n_threads)
#        res_list = p.map(self.rnr_frame, self.nd2_data)
#        for i_frame, frame in enumerate(res_list):
#            self.rnr_data[i_frame] = frame
#        print(f"RNR with {self.n_threads} threads completed.")
#        p.close()
#        return self.rnr_data
