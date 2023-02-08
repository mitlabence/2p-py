# TODO: extract these methods to a new python file, and move imports outside functions to speed up.
# taken from caiman.utils.visualization.py
def nb_view_patches_with_lfp_movement(Yr, A, C, b, f, d1, d2,
                                      t_lfp: np.array = None, y_lfp: np.array = None,
                                      t_mov: np.array = None, y_mov: np.array = None,
                                      YrA=None, image_neurons=None, thr=0.99, denoised_color=None, cmap='jet',
                                      r_values=None, SNR=None, cnn_preds=None):
    """
    Interactive plotting utility for ipython notebook

    Args:
        Yr: np.ndarray
            movie

        A,C,b,f: np.ndarrays
            outputs of matrix factorization algorithm

        d1,d2: floats
            dimensions of movie (x and y)

        YrA:   np.ndarray
            ROI filtered residual as it is given from update_temporal_components
            If not given, then it is computed (K x T)

        image_neurons: np.ndarray
            image to be overlaid to neurons (for instance the average)

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param r_values:
    """
    from past.utils import old_div
    import matplotlib as mpl
    from scipy.sparse import spdiags
    from caiman.utils.visualization import get_contours
    try:
        import bokeh
        import bokeh.plotting as bpl
        from bokeh.models import CustomJS, ColumnDataSource, Range1d, LabelSet
    except:
        print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")

    colormap = mpl.cm.get_cmap(cmap)
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    nA2 = np.ravel(np.power(A, 2).sum(0)) if isinstance(A, np.ndarray) else np.ravel(A.power(2).sum(0))
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
                       (A.T * np.matrix(Yr) -
                        (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                        A.T.dot(A) * np.matrix(C)) + C)
    else:
        Y_r = C + YrA

    x = np.arange(T)
    if image_neurons is None:
        image_neurons = A.mean(1).reshape((d1, d2), order='F')

    coors = get_contours(A, (d1, d2), thr)
    cc1 = [cor['coordinates'][:, 0] for cor in coors]
    cc2 = [cor['coordinates'][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    source = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))
    source_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

    code = """
            var data = source.data
            var data_ = source_.data
            var f = cb_obj.value - 1
            var x = data['x']
            var y = data['y']
            var y2 = data['y2']

            for (var i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+f*x.length]
                y2[i] = data_['z2'][i+f*x.length]
            }

            var data2_ = source2_.data;
            var data2 = source2.data;
            var c1 = data2['c1'];
            var c2 = data2['c2'];
            var cc1 = data2_['cc1'];
            var cc2 = data2_['cc2'];

            for (var i = 0; i < c1.length; i++) {
                   c1[i] = cc1[f][i]
                   c2[i] = cc2[f][i]
            }
            source2.change.emit();
            source.change.emit();
        """

    if r_values is not None:
        code += """
            var mets = metrics.data['mets']
            mets[1] = metrics_.data['R'][f].toFixed(3)
            mets[2] = metrics_.data['SNR'][f].toFixed(3)
            metrics.change.emit();
        """
        metrics = ColumnDataSource(data=dict(y=(3, 2, 1, 0),
                                             mets=('', "% 7.3f" % r_values[0], "% 7.3f" % SNR[0],
                                                   "N/A" if np.sum(cnn_preds) in (0, None) else "% 7.3f" % cnn_preds[
                                                       0]),
                                             keys=("Evaluation Metrics", "Spatial corr:", "SNR:", "CNN:")))
        if np.sum(cnn_preds) in (0, None):
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR))
        else:
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR, CNN=cnn_preds))
            code += """
                mets[3] = metrics_.data['CNN'][f].toFixed(3)
            """
        labels = LabelSet(x=0, y='y', text='keys', source=metrics, render_mode='canvas')
        labels2 = LabelSet(x=10, y='y', text='mets', source=metrics, render_mode='canvas', text_align="right")
        plot2 = bpl.figure(plot_width=200, plot_height=100, toolbar_location=None)
        plot2.axis.visible = False
        plot2.grid.visible = False
        plot2.tools.visible = False
        plot2.line([0, 10], [0, 4], line_alpha=0)
        plot2.add_layout(labels)
        plot2.add_layout(labels2)
    else:
        metrics, metrics_ = None, None

    callback = CustomJS(args=dict(source=source, source_=source_, source2=source2,
                                  source2_=source2_, metrics=metrics, metrics_=metrics_), code=code)

    plot = bpl.figure(plot_width=600, plot_height=200, x_range=Range1d(0, Y_r.shape[0]))
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1,
                  line_alpha=0.6, color=denoised_color)

    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr,
                       plot_width=int(min(1, d2 / d1) * 300),
                       plot_height=int(min(1, d1 / d2) * 300))

    plot1.image(image=[image_neurons[::-1, :]], x=0,
                y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1', 'c2', alpha=0.6, color='purple',
                line_width=2, source=source2)

    # create plot for lfp
    if y_lfp is not None:
        source_lfp = ColumnDataSource(data=dict(x=t_lfp, y=y_lfp))
        plot_lfp = bpl.figure(x_range=Range1d(t_lfp[0], t_lfp[-1]),
                              y_range=Range1d(y_lfp.min(), y_lfp.max()),
                              plot_width=plot.plot_width,
                              plot_height=plot.plot_height)
        plot_lfp.line("x", "y", source=source_lfp)
    # plot_mov = bpl.figure(x_range=xr, y_range=None)
    if y_mov is not None:
        source_mov = ColumnDataSource(data=dict(x=t_mov, y=y_mov))
        plot_mov = bpl.Figure(x_range=Range1d(t_mov[0], t_mov[-1]),
                              y_range=Range1d(y_mov.min(), y_mov.max()),
                              plot_width=plot.plot_width,
                              plot_height=plot.plot_height)
        plot_mov.line("x", "y", source=source_mov)
    if Y_r.shape[0] > 1:
        slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                                     title="Neuron Number")
        slider.js_on_change('value', callback)
        if y_mov is not None:
            if y_lfp is not None:  # both lfp and mov
                bpl.show(bokeh.layouts.layout([[slider], [bokeh.layouts.row(
                    plot1 if r_values is None else bokeh.layouts.column(plot1, plot2),
                    bokeh.layouts.column(plot, plot_lfp, plot_mov))]]))
            else:  # no lfp plot
                bpl.show(bokeh.layouts.layout([[slider], [bokeh.layouts.row(
                    plot1 if r_values is None else bokeh.layouts.column(plot1, plot2),
                    bokeh.layouts.column(plot, plot_mov))]]))
        else:  # no mov plot
            if y_lfp is not None:
                bpl.show(bokeh.layouts.layout([[slider], [bokeh.layouts.row(
                    plot1 if r_values is None else bokeh.layouts.column(plot1, plot2),
                    bokeh.layouts.column(plot, plot_lfp))]]))
            else:  # no lfp and no movement
                bpl.show(bokeh.layouts.layout([[slider], [bokeh.layouts.row(
                    plot1 if r_values is None else bokeh.layouts.column(plot1, plot2), plot)]]))
    else:
        bpl.show(bokeh.layouts.row(plot1 if r_values is None else
                                   bokeh.layouts.column(plot1, plot2), plot))

    return Y_r


# TODO: extract these methods to a new python file, and move imports outside functions to speed up.
# taken from caiman.utils.visualization.py
def nb_view_patches_manual_control(Yr, A, C, b, f, d1, d2,
                                   YrA=None, image_neurons=None, thr=0.99, denoised_color=None, cmap='jet',
                                   r_values=None, SNR=None, cnn_preds=None):
    """
    Interactive plotting utility for ipython notebook

    Args:
        Yr: np.ndarray
            movie

        A,C,b,f: np.ndarrays
            outputs of matrix factorization algorithm

        d1,d2: floats
            dimensions of movie (x and y)

        YrA:   np.ndarray
            ROI filtered residual as it is given from update_temporal_components
            If not given, then it is computed (K x T)

        image_neurons: np.ndarray
            image to be overlaid to neurons (for instance the average)

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param r_values:
    """
    from past.utils import old_div
    import matplotlib as mpl
    from scipy.sparse import spdiags
    from caiman.utils.visualization import get_contours
    try:
        import bokeh
        import bokeh.plotting as bpl
        from bokeh.models import CustomJS, ColumnDataSource, Range1d, LabelSet
        from bokeh.models.widgets.buttons import Button, Toggle
    except:
        print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")

    # TODO: idx_components and idx_components_bad refer to indices of accepted/rejected neurons, use these in
    #  nb_view_components_manual_control. If These don't exist, that means select_components has been called... I don't
    #  know if it is still possible (easily) to move the neurons from one group to the other.
    REJECTED_COLOR = "red"
    REJECTED_TEXT = "rejected"
    ACCEPTED_COLOR = "green"
    ACCEPTED_TEXT = "accepted"

    colormap = mpl.cm.get_cmap(cmap)
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    nA2 = np.ravel(np.power(A, 2).sum(0)) if isinstance(A, np.ndarray) else np.ravel(A.power(2).sum(0))
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
                       (A.T * np.matrix(Yr) -
                        (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                        A.T.dot(A) * np.matrix(C)) + C)
    else:
        Y_r = C + YrA

    x = np.arange(T)
    if image_neurons is None:
        image_neurons = A.mean(1).reshape((d1, d2), order='F')

    coors = get_contours(A, (d1, d2), thr)
    cc1 = [cor['coordinates'][:, 0] for cor in coors]
    cc2 = [cor['coordinates'][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    source = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))
    source_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

    code = """
            var data = source.data
            var data_ = source_.data
            var f = cb_obj.value - 1
            var x = data['x']
            var y = data['y']
            var y2 = data['y2']

            for (var i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+f*x.length]
                y2[i] = data_['z2'][i+f*x.length]
            }

            var data2_ = source2_.data;
            var data2 = source2.data;
            var c1 = data2['c1'];
            var c2 = data2['c2'];
            var cc1 = data2_['cc1'];
            var cc2 = data2_['cc2'];

            for (var i = 0; i < c1.length; i++) {
                   c1[i] = cc1[f][i]
                   c2[i] = cc2[f][i]
            }
            source2.change.emit();
            source.change.emit();
        """

    if r_values is not None:
        code += """
            var mets = metrics.data['mets']
            mets[1] = metrics_.data['R'][f].toFixed(3)
            mets[2] = metrics_.data['SNR'][f].toFixed(3)
            metrics.change.emit();
        """
        metrics = ColumnDataSource(data=dict(y=(3, 2, 1, 0),
                                             mets=('', "% 7.3f" % r_values[0], "% 7.3f" % SNR[0],
                                                   "N/A" if np.sum(cnn_preds) in (0, None) else "% 7.3f" % cnn_preds[
                                                       0]),
                                             keys=("Evaluation Metrics", "Spatial corr:", "SNR:", "CNN:")))
        if np.sum(cnn_preds) in (0, None):
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR))
        else:
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR, CNN=cnn_preds))
            code += """
                mets[3] = metrics_.data['CNN'][f].toFixed(3)
            """
        labels = LabelSet(x=0, y='y', text='keys', source=metrics, render_mode='canvas')
        labels2 = LabelSet(x=10, y='y', text='mets', source=metrics, render_mode='canvas', text_align="right")
        plot2 = bpl.figure(plot_width=200, plot_height=100, toolbar_location=None)
        plot2.axis.visible = False
        plot2.grid.visible = False
        plot2.tools.visible = False
        plot2.line([0, 10], [0, 4], line_alpha=0)
        plot2.add_layout(labels)
        plot2.add_layout(labels2)
    else:
        metrics, metrics_ = None, None

    callback = CustomJS(args=dict(source=source, source_=source_, source2=source2,
                                  source2_=source2_, metrics=metrics, metrics_=metrics_), code=code)

    plot = bpl.figure(plot_width=600, plot_height=200, x_range=Range1d(0, Y_r.shape[0]))
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1,
                  line_alpha=0.6, color=denoised_color)

    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr,
                       plot_width=int(min(1, d2 / d1) * 300),
                       plot_height=int(min(1, d1 / d2) * 300))

    plot1.image(image=[image_neurons[::-1, :]], x=0,
                y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1', 'c2', alpha=0.6, color='purple',
                line_width=2, source=source2)

    original_status = Button(label="original: " + ACCEPTED_TEXT, disabled=True, width=150, background=ACCEPTED_COLOR)
    current_status = Button(label="current: " + REJECTED_TEXT, disabled=True, width=150, background=REJECTED_COLOR)
    transfer_button = Button(label="Transfer", width=100)
    save_button = Button(label="Save changes", width=100)
    if Y_r.shape[0] > 1:
        slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                                     title="Neuron Number")
        slider.js_on_change('value', callback)
        bpl.show(bokeh.layouts.layout([[slider, transfer_button, original_status, current_status], [bokeh.layouts.row(
            plot1 if r_values is None else bokeh.layouts.column(plot1, plot2), plot)]]))
    else:
        bpl.show(bokeh.layouts.row(plot1 if r_values is None else
                                   bokeh.layouts.column(plot1, plot2), plot))
    # TODO: using slider should emit changes to original_status and current_status. Use subscribed_events
    # TODO: also pressing transfer should update current_status.
    # ButtonClick as event (for transfering)

    # TODO: on clicking Transfer, print something; then print current slider value
    # TODO: apparently, changing the python variables directly is not possible. Can create a JS variable initially, a
    #  list of accepted/rejected. Create two of these, one for original status, the other vill be modified (current
    #  status). Pressing Transfer changes the current status. Upon clicking on another button, export the old and
    #  current status lists into python variables, then do the conversion in python."
    return Y_r


def nb_view_components_with_lfp_movement(estimates,
                                         t_lfp: np.array = None, y_lfp: np.array = None,
                                         t_mov: np.array = None, y_mov: np.array = None,
                                         Yr=None, img=None, idx=None, denoised_color=None, cmap='jet', thr=0.99):
    """view spatial and temporal components interactively in a notebook, along with LFP and movement

    Args:
        estimates : the estimates attribute of a CNMF instance
        t_lfp: np.ndarray
            time data of lfp recording
        y_lfp: np.ndarray
            amplitude of lfp recording
        t_mov: np.ndarray
            time data of movement recording
        y_mov: np.ndarray
            amplitude of movement recording
        Yr :    np.ndarray
            movie in format pixels (d) x frames (T)

        img :   np.ndarray
            background image for contour plotting. Default is the mean
            image of all spatial components (d1 x d2)

        idx :   list
            list of components to be plotted

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param estimates:
    """
    from matplotlib import pyplot as plt
    import scipy

    if 'csc_matrix' not in str(type(estimates.A)):
        estimates.A = scipy.sparse.csc_matrix(estimates.A)

    plt.ion()
    nr, T = estimates.C.shape
    if estimates.R is None:
        estimates.R = estimates.YrA
    if estimates.R.shape != [nr, T]:
        if estimates.YrA is None:
            estimates.compute_residuals(Yr)
        else:
            estimates.R = estimates.YrA

    if img is None:
        img = np.reshape(np.array(estimates.A.mean(axis=1)), estimates.dims, order='F')

    if idx is None:
        nb_view_patches_with_lfp_movement(
            Yr, estimates.A, estimates.C, estimates.b, estimates.f, estimates.dims[0], estimates.dims[1],
            t_lfp=t_lfp, y_lfp=y_lfp, t_mov=t_mov, y_mov=y_mov,
            YrA=estimates.R, image_neurons=img, thr=thr, denoised_color=denoised_color, cmap=cmap,
            r_values=estimates.r_values, SNR=estimates.SNR_comp, cnn_preds=estimates.cnn_preds)
    else:
        nb_view_patches_with_lfp_movement(
            Yr, estimates.A.tocsc()[:, idx], estimates.C[idx], estimates.b, estimates.f,
            estimates.dims[0], estimates.dims[1], t_lfp=t_lfp, y_lfp=y_lfp, t_mov=t_mov, y_mov=y_mov,
            YrA=estimates.R[idx], image_neurons=img,
            thr=thr, denoised_color=denoised_color, cmap=cmap,
            r_values=None if estimates.r_values is None else estimates.r_values[idx],
            SNR=None if estimates.SNR_comp is None else estimates.SNR_comp[idx],
            cnn_preds=None if np.sum(estimates.cnn_preds) in (0, None) else estimates.cnn_preds[idx])
    return estimates


def nb_view_components_manual_control(estimates,
                                      Yr=None, img=None, idx=None, denoised_color=None, cmap='jet', thr=0.99,
                                      mode: str = "reject"):
    """view spatial and temporal components interactively in a notebook

    Args:
        estimates : the estimates attribute of a CNMF instance
        t_lfp: np.ndarray
            time data of lfp recording
        y_lfp: np.ndarray
            amplitude of lfp recording
        t_mov: np.ndarray
            time data of movement recording
        y_mov: np.ndarray
            amplitude of movement recording
        Yr :    np.ndarray
            movie in format pixels (d) x frames (T)

        img :   np.ndarray
            background image for contour plotting. Default is the mean
            image of all spatial components (d1 x d2)

        idx :   list
            list of components to be plotted

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param estimates:

		mode: string
			Whether to go through accepted components and reject manually ("accepted" or reject"), or go through rejected components and move manually to accepted ("rejected" or "accept").
    """
    from matplotlib import pyplot as plt
    import scipy
    # TODO: if refit is used, estimates.idx_components and idx_components_bad are empty (None). Need to still plot
    #  these as all accepted
    if 'csc_matrix' not in str(type(estimates.A)):
        estimates.A = scipy.sparse.csc_matrix(estimates.A)

    plt.ion()
    nr, T = estimates.C.shape
    if estimates.R is None:
        estimates.R = estimates.YrA
    if estimates.R.shape != [nr, T]:
        if estimates.YrA is None:
            estimates.compute_residuals(Yr)
        else:
            estimates.R = estimates.YrA

    if img is None:
        img = np.reshape(np.array(estimates.A.mean(axis=1)), estimates.dims, order='F')

    if idx is None:
        nb_view_patches_manual_control(
            Yr, estimates.A, estimates.C, estimates.b, estimates.f, estimates.dims[0], estimates.dims[1],
            YrA=estimates.R, image_neurons=img, thr=thr, denoised_color=denoised_color, cmap=cmap,
            r_values=estimates.r_values, SNR=estimates.SNR_comp, cnn_preds=estimates.cnn_preds)
    else:
        nb_view_patches_manual_control(
            Yr, estimates.A.tocsc()[:, idx], estimates.C[idx], estimates.b, estimates.f,
            estimates.dims[0], estimates.dims[1],
            YrA=estimates.R[idx], image_neurons=img,
            thr=thr, denoised_color=denoised_color, cmap=cmap,
            r_values=None if estimates.r_values is None else estimates.r_values[idx],
            SNR=None if estimates.SNR_comp is None else estimates.SNR_comp[idx],
            cnn_preds=None if np.sum(estimates.cnn_preds) in (0, None) else estimates.cnn_preds[idx])
    return estimates