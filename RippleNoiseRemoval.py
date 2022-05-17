import pims_nd2
import numpy as np
from matplotlib import pyplot as plt

nd2_fpath = 'D:/PhD/Data/T386_MatlabTest/T386_20211202_green.nd2'
win = 40
end_x = 0
end_y = 0
# now nd2_to_caiman.py


def np_arr_from_nd2(nd2_fpath: str):
    # set iter_axes to "t"
    # then: create nd array with sizes matching frame size,
    with pims_nd2.ND2_Reader(nd2_fpath) as nikon_file:  # todo: get metadata too?

        sizes_dict = nikon_file.sizes
        sizes = (sizes_dict['t'], sizes_dict['x'], sizes_dict['y'])

        # dtype would be float32 by default...
        frames_arr = np.zeros(sizes, dtype=nikon_file.pixel_type)

        # TODO: probably it is not even necessary to export an np.array, as nikon_file is an iterable of
        #  subclasses of np array... not sure what caiman needs
        for i_frame, frame in enumerate(nikon_file):
            # not sure if dtype needed here
            frames_arr[i_frame] = np.array(
                nikon_file[0], dtype=nikon_file.pixel_type)
        return frames_arr


nd2_data = np_arr_from_nd2(nd2_fpath)


"""
# optional: show original figure of which plot is made
fig = plt.figure(figsize=(18,18))  # figsize does not work...
plt.pcolormesh(nd2_data[0])
plt.show()
"""

freq_matrix = np.fft.fftshift(np.fft.fft2(nd2_data[0, :, :]))
amplitude_image = np.log(np.abs(freq_matrix))
bright_spikes = amplitude_image > 10.8  # default amplitude threshold
rectangle_filter_boundary = np.zeros(amplitude_image.shape)
filtered_spikes = np.copy(bright_spikes)

end_y = amplitude_image.shape[0]  # todo: maybe switched?
end_x = amplitude_image.shape[1]

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


# plot various data
fig, axs = plt.subplots(2, 3, figsize=(20, 20))
axs[0, 0].pcolormesh(amplitude_image)
axs[0, 1].pcolormesh(bright_spikes + rectangle_filter_boundary)
axs[0, 2].pcolormesh(np.abs(freq_filtered))
axs[1, 0].pcolormesh(filtered_spikes)
for i_row in range(len(amplitude_image)):
    axs[1, 1].plot(amplitude_image[i_row, :])
# axs[0,0].colorbar()
axs[1, 2].pcolormesh(np.log(np.abs(freq_matrix - freq_filtered)))
plt.show()
