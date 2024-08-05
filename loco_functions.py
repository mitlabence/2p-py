import numpy as np
import warnings


def apply_threshold(speed_trace, episodes, temporal_threshold, amplitude_threshold):
    """
    Given a speed trace and a list of tuples (i_begin_frame, i_end_frame) marking running episodes, this function discards those that
    a.) are shorter than the defined temporal threshold (in units of frames),
    OR
    b.) the amplitude of the absolute trace does not reach the amplitude threshold during the episode.
    Returns the filtered episodes.
    """

    discard_list = []
    # tuple of (i_begin, i_end). Assume [i_begin:i_end+1] is correct, see get_episodes()
    for i_episode, episode in enumerate(episodes):
        episode_trace = speed_trace[episode[0]:episode[1]+1]
        # filter by temporal threshold
        if len(episode_trace) < temporal_threshold:
            # print(f"{len(episode_trace)}")
            if i_episode not in discard_list:
                discard_list.append(i_episode)
        # filter by amplitude threshold
        if max(np.abs(episode_trace)) < amplitude_threshold:
            if i_episode not in discard_list:
                discard_list.append(i_episode)
    discard_list = sorted(discard_list)

    # discard components
    episodes_filtered = [episodes[i]
                         for i in range(len(episodes)) if i not in discard_list]
    return episodes_filtered


def get_episodes(segment, merge_episodes=False, merge_threshold_frames=8, return_begin_end_frames=False):
    """Given a binary trace (0 is rest, 1 is locomotion), return the (beginning, end) frames of each locomotion episode.
    If merge_episodes=True, also  

    Parameters
    ----------
    segment : _type_
        _description_
    merge_episodes : bool, optional
        _description_, by default False
    merge_threshold_frames : int, optional
        _description_, by default 8
    return_begin_end_frames : bool, optional
        Whether to return the number of episodes, or the episode beginning and end frames. 
        If set to return indices (True), then (i_begin, i_end) are both inclusive in 0-indexing! By default False

    Returns
    -------
    _type_
        _description_
    """
    #

    n_eps = 0
    episode_lengths = []  # in frame units
    episodes = []
    n_episodes = 0
    current_episode_len = 0

    episode_begin = 0
    episode_end = 0

    # algorithm: detect episode begin and episode end. record it in list

    # check current and next element for end of a episode: ...100...
    for i_frame in range(len(segment)-1):
        if segment[i_frame] == 1:  # current frame is part of an episode
            # increase current episode length
            # check if beginning of an episode or segment starts with an episode
            if i_frame == 0 or segment[i_frame - 1] == 0:
                episode_begin = i_frame
            current_episode_len += 1
            if segment[i_frame+1] == 0:  # episode ends with next frame
                n_episodes += 1
                episode_lengths.append(current_episode_len)
                episodes.append((episode_begin, i_frame))
                current_episode_len = 0
    if segment[-1] == 1:  # check if there is one episode that does not end
        n_episodes += 1
        # add last segment to segments list
        current_episode_len += 1
        episode_lengths.append(current_episode_len)
        episodes.append((episode_begin, len(segment)-1))
        current_episode_len = 0

    assert current_episode_len == 0
    if merge_episodes:
        if len(episodes) < 2:  # single (or zero) episode cannot be merged
            if return_begin_end_frames:
                return episodes
            else:
                return [ep[1]-ep[0] + 1 for ep in episodes]

        # merge episodes that are close to each other
        episodes_merged = []

        episode_begin = episodes[0][0]
        episode_end = episodes[0][1]
        # starting with second episode, check if current episode can be merged with previous. If yes, update episode_end.
        # If not, add previous episode to list, update episode_begin and episode_end to current episode values

        for i_episode in range(1, len(episodes)):
            current_episode_begin = episodes[i_episode][0]
            current_episode_end = episodes[i_episode][1]

            delta = current_episode_begin - episode_end

            if delta <= merge_threshold_frames:  # merge current episode to previous one
                episode_end = current_episode_end
            else:  # add previous episode to list, start with current episode
                episodes_merged.append((episode_begin, episode_end))
                episode_begin = current_episode_begin
                episode_end = current_episode_end
        # add last segment to list
        episodes_merged.append((episode_begin, episode_end))
        if return_begin_end_frames:
            return episodes_merged
        else:
            episode_lengths_merged = [
                ep[1]-ep[0] + 1 for ep in episodes_merged]
            return episode_lengths_merged
    if return_begin_end_frames:
        return episodes
    else:
        return episode_lengths  # len() shows n_episodes


def calculate_avg_speed(speed_trace):
    """Given a speed trace list or numpy array, calculate the average absolute speed. Resting periods where speed=0 are ignored.

    Parameters
    ----------
    speed_trace : iterable, list[float] or np.array(float)
        A 1D list or numpy array of speed values (float)

    Returns
    -------
    float
        The average apsolute speed over the whole trace
    """
    speed_trace = np.array(speed_trace)
    return np.mean(np.abs(speed_trace[speed_trace > 0]))


def calculate_max_speed(speed_trace):
    """Given a speed trace list or numpy array, calculate the absolute maximum speed.

    Parameters
    ----------
    speed_trace : iterable, list[float] or np.array(float)
        A 1D list or numpy array of speed values (float)
    Returns
    -------
    float
        The maximum absolute speed. This can be the absolute value of a negative speed reached as well!
    """
    speed_trace = np.array(speed_trace)
    # np.median(np.sort(speed_trace)[floor(0.95*len(speed_trace)):])
    return np.max(np.abs(speed_trace))


def get_trace_delta(trace, i_begin, i_end_exclusive):
    """Given a monotonously changing trace, get the change during the segment starting at frame i_begin, and ending one frame before i_end_exclusive.
    I.e. returns trace[i_begin:i_end_exclusive] [-1] - [0].
    Parameters
    ----------
    trace : 1D iterable
        The monotonously changing complete trace
    i_begin : int
        The first (0-indexed) frame to include.
    i_end_exclusive : int
        One after the last (0-indexed) frame to include. Reflects a[x:y] indexing conventions.
    Returns
    -------
    float
        The change during the 
    """
    trace = np.array(trace)
    if not (np.all(trace[1:] >= trace[:-1]) or np.all(trace[1:] <= trace[:-1])):
        warnings.warn("get_trace_delta(): trace is not monotonous!")
    trace_cut = trace[i_begin:i_end_exclusive]
    return trace_cut[-1] - trace_cut[0]
