import h5py
import sys
sys.path.insert(1, 'D:\\PhD\\2p-py')


from labrotation.two_photon_session import nb_view_patches_with_lfp_movement, TwoPhotonSession
import matplotlib.pylab as plt
import subprocess


FLUO_COLOR = "green"
LFP_COLOR = "black"
LOCO_COLOR = "grey"
BAR_COLOR = "red"

matlab_2p_folder = "D:\\PhD\\matlab-2p"

nd2_fpath = "D:\\testdata\\T386_ChR2_LR_elec_d6_20211202_488nm_500msec_002.nd2"
nd2_timestamps_fpath = "D:\\testdata\\T386.021221.1201_nik.txt"
labview_fpath = "D:\\testdata\\T386.021221.1201.txt"
labview_timestamps_fpath = "D:\\testdata\\T386.021221.1201time.txt"
lfp_fpath = "D:\\testdata\\21d02001.abf"

result_fpath = "D:\\testdata\\out_test.mp4"

tps = TwoPhotonSession.init_and_process(nd2_fpath, nd2_timestamps_fpath, labview_fpath, labview_timestamps_fpath, lfp_fpath, matlab_2p_folder)
fluor = tps.return_nikon_mean()
t = tps.belt_dict["time"]
t_fluor = tps.belt_dict["tsscn"]
t_fluor_div = t_fluor/1000.
loco = tps.belt_scn_dict["speed"] # tps.belt_dict["speed"]
t_lfp, y_lfp = tps.lfp_lfp()
lfp_y = tps.lfp_df_cut["y_lfp"]
lfp_t = tps.lfp_df_cut["t_lfp_corrected"]
loco_y = tps.lfp_df_cut["y_mov"]
loco_t = tps.lfp_df_cut["t_mov_corrected"]
nik = tps.get_nikon_data()


def saveVid(i_frame_begin, i_frame_end):  # in 1-indexing
    t_vertline = t_fluor[i_frame_begin - 1]
    VERTBAR_HEIGHT = 8.  # adjust this based on gridspec height_ratios and other stuff...
    # For equal-sized three plots, 3.5 used to be enough. For 3:3:1, 8. was needed

    fig = plt.figure(figsize=(18, 10))
    canvas_width, canvas_height = fig.canvas.get_width_height()
    gridspec = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1])
    ax_fluor = fig.add_subplot(gridspec[0, 0])
    ax_lfp = fig.add_subplot(gridspec[1, 0])
    ax_loco = fig.add_subplot(gridspec[2, 0])
    ax_nikon = fig.add_subplot(gridspec[:, 1])

    ax_fluor.plot(t_fluor / 1000., fluor, color=FLUO_COLOR)
    ax_lfp.plot(lfp_t, lfp_y, color=LFP_COLOR, linewidth=0.3)
    # ax_loco.plot(loco_t, loco_y, color="red")
    ax_loco.plot(t_fluor_div, loco, color=LOCO_COLOR)

    fig_nikon = ax_nikon.imshow(nik[0])

    # vline_fluor = ax_fluor.axvline(x=t_vertline, ymin=-1.2, ymax=1, zorder=0,clip_on=False, color="black")
    # vline_lfp = ax_lfp.axvline(x=t_vertline, ymin=-1.2, ymax=1, zorder=0,clip_on=False, color="black")
    vline_loco = ax_loco.axvline(x=t_vertline, ymin=0, ymax=VERTBAR_HEIGHT, zorder=0, clip_on=False, color=BAR_COLOR)

    ax_fluor.set_xlim([t_fluor_div[i_frame_begin - 5], t_fluor_div[i_frame_end + 5]])
    ax_lfp.set_xlim([t_fluor_div[i_frame_begin - 5], t_fluor_div[i_frame_end + 1]])
    ax_loco.set_xlim([t_fluor_div[i_frame_begin - 5], t_fluor_div[i_frame_end + 1]])

    ax_fluor.set_ylim([0, 150])
    ax_lfp.set_ylim([-0.7, 0.25])
    ax_loco.set_ylim([-0.12, 0.12])

    ax_nikon.axis('off')
    ax_fluor.axis('off')
    ax_lfp.axis('off')
    ax_loco.spines['left'].set_visible(False)
    ax_loco.spines['right'].set_visible(False)
    ax_loco.spines['top'].set_visible(False)
    ax_loco.get_yaxis().set_ticks([])

    ax_loco.tick_params(axis='x', labelsize=18)
    ax_loco.set_xlabel("Time (s)", fontsize=20)

    def update(frame):
        # your matplotlib code goes here
        fig_nikon.set_data(nik[frame])

        vline_loco.set_data([t_fluor_div[frame - 1], t_fluor_div[frame - 1]], [0, VERTBAR_HEIGHT])

    # Open an ffmpeg process
    # lossless encoding:
    # https://stackoverflow.com/questions/37344997/how-to-get-a-lossless-encoding-with-ffmpeg-libx265
    cmdstring = ('ffmpeg',
                 '-y', '-r', '15',  # overwrite, 1fps
                 '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
                 '-pix_fmt', 'argb',  # format
                 '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
                 # '-vcodec', 'mpeg4', result_fpath) # use mpeg4 encoding
                 '-c:v', 'libx265',
                 '-x265-params', '"profile=monochrome12:crf=0:lossless=1:preset=veryslow:qp=0"',
                 result_fpath)
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=True)

    # Draw frames and write to the pipe
    for frame in range(i_frame_begin, i_frame_end + 1):
        print(frame)
        # draw the frame
        update(frame)
        fig.canvas.draw()

        # extract the image as an ARGB string
        string = fig.canvas.tostring_argb()
        # write to pipe
        p.stdin.write(string)

    # Finish up
    p.communicate()


saveVid(4450, 4700)