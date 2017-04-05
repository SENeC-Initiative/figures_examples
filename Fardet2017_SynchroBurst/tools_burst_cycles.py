#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tools to get the details of the bursting cycles """

import nest
import numpy as np

from ActivStudy.analysis import find_idx_nearest


'''
Data processing
----------------

Define the functions that will be used to delimitate the bursting cycles.
'''

def _get_cycle_extremity(ts, Vs, ws, time_min, time_max, Vth, num_neurons):
    '''
    Return the index at which the attractor reaches one of its extremity (start
    or end), for:
        V = (3Vth + V_down)/4
    and
        w < w_down

    Returns
    -------
    idx_cycle_extremity : index of this extremity
    '''
    # time delimitation
    idx_start = find_idx_nearest(ts, time_min)
    idx_end = find_idx_nearest(ts, time_max)
    # idx V_min and value V_start / w_down
    idx_min = idx_start + np.argmin(Vs[idx_start:idx_end])
    V_start = (3*Vth*num_neurons + Vs[idx_min])/4 # non averaged Vs
    w_down = ws[idx_min]
    # index of cycle extremity
    idx_cycle_extremity = idx_min \
        + np.where(np.isclose(Vs[idx_min:], V_start, 0.001)
                   * (ws[idx_min:] < w_down))[0][0]
    return idx_cycle_extremity


def get_data(activ_object, first_burst_number, last_burst_number,
             num_neurons, Vth):
    '''
    Get the data from a ``Simulator`` instance.

    Trim it to keep only the points belonging to the chosen interval, which is
    the data starting immediately after the `first_burst_number`th burst:

    * the following interburst
    * the successive bursts (until `last_burst_number` included)

    Parameters
    ----------
    activ_object : ``Simulator`` instance.
    first_burst_number : int
        Number of the burst after which data will be processed.
    last_burst_number : int
        Number of the burst after which data will stop being processed.
    num_neurons : int
        The number of neurons.
    Vth : float
        Value of the threshold potential.
    '''
    Vs, ws = None, None
    time_start = activ_object.phases["bursting"][first_burst_number-1][1]
    time_poststart = activ_object.phases["bursting"][first_burst_number][1]
    time_preend = activ_object.phases["bursting"][last_burst_number][1]
    time_end = activ_object.phases["bursting"][last_burst_number+1][1]
    for recorder, record in zip(activ_object.recorders, activ_object.record):
        if "V_m" in record:
            data = nest.GetStatus(recorder, "events")[0]
            # get extremities of the cycle
            idx_cycle_start = _get_cycle_extremity(
                data["times"], data["V_m"], data["w"], time_start,
                time_poststart, Vth, num_neurons)
            idx_cycle_end = _get_cycle_extremity(
                data["times"], data["V_m"], data["w"], time_preend, time_end,
                Vth, num_neurons)
            # keep only relevant data
            Vs = data["V_m"][idx_cycle_start:idx_cycle_end] / num_neurons
            ws = data["w"][idx_cycle_start:idx_cycle_end] / num_neurons
    if Vs is None:
        raise AttributeError("""`activ_object` did not contain any record of
                             the state parameters.""")
    return Vs, ws
