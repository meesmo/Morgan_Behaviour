#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23.8.14 code for analyzing classical conditioning data on new 2023 rigs.

@author: michaellynn
"""

import copy
import numpy as np
import h5py


def find_lick_onsets(v_lick_raw, t_lick_raw,
                     cutoff_lick_freq=15):
    """Finds lick onsets given a raw voltage and time signal.
    Both voltage and time must be single vectors (ie one trial)
    and t_lick_raw should be in seconds.

    Parameters
    --------------
    v_lick_raw : np.ndarray, dtype=bool
        A one-dimensional array of lick voltages (boolean).

    t_lick_raw : np.ndarray, dtype=float
        A one-dimensional array of lick times.

    cutoff_lick_freq : float
        The maximum lick frequency permitted to be registered, in Hz.
        Useful if the lickometer detects one lick as multiple (error)
        which would appear as a transient, very high lickrate.
        (Mice typically have a mean lickrate of 7-8Hz although this can
        increase very slightly during licking bursts.)

    Returns
    -------------
    t_licks : np.ndarray
        An array of lick onset times, in seconds.
    """

    cutoff_lick_isi = 1/cutoff_lick_freq
    ind_lick_onset = np.where(np.diff(v_lick_raw) > 0.5)[0]

    t_licks = []
    for _count, ind_lick in enumerate(ind_lick_onset):
        _t_lick = t_lick_raw[ind_lick]
        if _count > 0:
            if _t_lick - t_licks[-1] > cutoff_lick_isi:
                t_licks.append(_t_lick)
        elif _count == 0:
            t_licks.append(_t_lick)

    t_licks = np.array(t_licks)

    return t_licks


def calc_lickrate_between(t_lick, lower_bound, upper_bound):
    """Calculates a mean lickrate between two time boundaries,
    given a vector of lick times.

    Parameters
    -------------
    t_lick : np.ndarray
        An array of lick times, in seconds.

    lower_bound : float
        The lower bound of accepted times, in seconds.

    upper_bound : float
        The upper bound of accepted times, in seconds.

    Returns
    ------------
    lickrate : float
        Rate of licking, over the designated time interavl, in Hz
    """

    t_lick_filt = t_lick[np.logical_and(
        t_lick > lower_bound, t_lick < upper_bound)]
    lickrate = len(t_lick_filt) / (upper_bound - lower_bound)
    return lickrate


class BehaviorData(object):
    def __init__(self, fname):
        """Loads an hdf5 file and parses information.

        Parameters
        ------------
        fname : string
            The filename of the hdf5 file
        """
        f = h5py.File(fname)
        print(list(f.keys()))

        self.n_trials = f['lick_r']['volt'].shape[0]
        self.rew_vol = np.array(f['rew_r']['volume'])

        self.t_rew = np.array(f['rew_r']['t']) / 1000
        self.t_tone_start = np.array(f['sample_tone']['t']) / 1000
        self.t_tone_end = np.array(f['sample_tone']['end']) / 1000
        self.t_trial_end = (np.array(f['t_end']) - np.array(f['t_start'])) \
            / 1000

        self._licks_v = np.array(f['lick_r']['volt'])
        self._licks_t = np.array(f['lick_r']['t'])

        # convert raw lick signal to lick onsets
        self.licks = np.empty(self.n_trials, dtype=np.ndarray)

        for trial in range(self.n_trials):
            self.licks[trial] = find_lick_onsets(self._licks_v[trial],
                                                 self._licks_t[trial]/1000)

    def add_licking_stats(self, bl_normalize=True):
        """Adds lick rates for each trial and epoch.

        Parameters
        -------------
        bl_normalize: bool
            Specifies whether to normalize each trial's lickrate
            by the baseline lick rate (True) or keep raw lickrates (False).
        """
        _templ = np.empty(self.n_trials)
        self._epochs = ['bl', 'tone', 'trace', 'rew']

        self.lickrate = {'bl': copy.deepcopy(_templ),
                         'bl_raw': copy.deepcopy(_templ),
                         'tone': copy.deepcopy(_templ),
                         'trace': copy.deepcopy(_templ),
                         'rew': copy.deepcopy(_templ)}

        for trial in range(self.n_trials):
            _t_tone_start = self.t_tone_start[trial]
            _t_tone_end = self.t_tone_end[trial]
            _t_rew = self.t_rew[trial]
            _t_trial_end = self.t_trial_end[trial]

            _lrate_bl = calc_lickrate_between(
                self.licks[trial], 0, _t_tone_start)
            _lrate_tone = calc_lickrate_between(
                self.licks[trial], _t_tone_start, _t_tone_end)
            _lrate_trace = calc_lickrate_between(
                self.licks[trial], _t_tone_end, _t_rew)
            _lrate_rew = calc_lickrate_between(
                self.licks[trial], _t_rew, _t_trial_end)

            self.lickrate['bl_raw'][trial] = _lrate_bl

            if bl_normalize is True:
                _lrate_bl -= _lrate_bl
                _lrate_tone -= _lrate_bl
                _lrate_trace -= _lrate_bl
                _lrate_rew -= _lrate_bl

            self.lickrate['bl'][trial] = _lrate_bl
            self.lickrate['tone'][trial] = _lrate_tone
            self.lickrate['trace'][trial] = _lrate_trace
            self.lickrate['rew'][trial] = _lrate_rew

        return

    def print_mean_lickrates(self):
        """Prints the mean lickrates, divided by trial type,
        for each epoch of the trial.
        """
        print('mean lickrates printing...')
        _rew_vol_unique = np.unique(self.rew_vol)

        # iterate through unique reward volumes and print stats
        for _rew_vol_ind in _rew_vol_unique:
            print(f'rew vol: {_rew_vol_ind}uL')
            _bool_mask = self.rew_vol == _rew_vol_ind

            _lrate_bl = np.mean(self.lickrate['bl'][_bool_mask])
            _lrate_bl_raw = np.mean(self.lickrate['bl_raw'][_bool_mask])
            _lrate_tone = np.mean(self.lickrate['bl'][_bool_mask])
            _lrate_trace = np.mean(self.lickrate['trace'][_bool_mask])
            _lrate_rew = np.mean(self.lickrate['rew'][_bool_mask])

            print(f'\tbaseline: {_lrate_bl:.2f}Hz')
            print(f'\tbaseline raw: {_lrate_bl_raw:.2f}Hz')
            print(f'\ttone: {_lrate_tone:.2f}Hz')
            print(f'\ttrace: {_lrate_trace:.2f}Hz')
            print(f'\treward: {_lrate_rew:.2f}Hz')

        return

    def print_range_lickrates(self):
        """Prints the range of lickrates, divided by trial type,
        for each epoch of the trial.
        """
        print('lickrate ranges printing...')
        _rew_vol_unique = np.unique(self.rew_vol)

        # iterate through unique reward volumes and print stats
        for _rew_vol_ind in _rew_vol_unique:
            print(f'rew vol: {_rew_vol_ind}uL')
            _bool_mask = self.rew_vol == _rew_vol_ind

            _lrate_bl_min = np.min(self.lickrate['bl'][_bool_mask])
            _lrate_bl_raw_min = np.min(self.lickrate['bl_raw'][_bool_mask])
            _lrate_tone_min = np.min(self.lickrate['bl'][_bool_mask])
            _lrate_trace_min = np.min(self.lickrate['trace'][_bool_mask])
            _lrate_rew_min = np.min(self.lickrate['rew'][_bool_mask])

            _lrate_bl_max = np.max(self.lickrate['bl'][_bool_mask])
            _lrate_bl_raw_max = np.max(self.lickrate['bl_raw'][_bool_mask])
            _lrate_tone_max = np.max(self.lickrate['bl'][_bool_mask])
            _lrate_trace_max = np.max(self.lickrate['trace'][_bool_mask])
            _lrate_rew_max = np.max(self.lickrate['rew'][_bool_mask])

            print(f'\tbaseline: {_lrate_bl_min:.2f}-{_lrate_bl_max:.2f}Hz')
            print(f'\tbaseline raw: {_lrate_bl_raw_min:.2f}-{_lrate_bl_raw_max:.2f}Hz')
            print(f'\ttone: {_lrate_tone_min:.2f}-{_lrate_tone_max:.2f}Hz')
            print(f'\ttrace: {_lrate_trace_min:.2f}-{_lrate_trace_max:.2f}Hz')
            print(f'\treward: {_lrate_rew_min:.2f}-{_lrate_rew_max:.2f}Hz')

        return


# -------------------------

b = BehaviorData('ms8012_2023-08-02_block1.hdf5')
b.add_licking_stats()
b.print_mean_lickrates()
b.print_range_lickrates()
