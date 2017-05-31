#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot the time evolution of the theoretical mean-field model """

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = False
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import numpy as np
from scipy.signal import argrelmax

import nest
from PyNeurActiv.models import Fardet2017_SynchroBurst, Simulator_SynchroBurst


'''
Parameters for the network.

Neuronal parameters are imported from parameters.py, as well as average degree
and number of neurons.
Delay, synaptic model and other specific parameters are set directly.
'''

from parameters import di_param, avg_deg, num_neurons

delay = 1.

di_param["weight"] = 70.
di_param["avg_deg"] = avg_deg
di_param["delay"] = delay

model = "alpha"
steps = 1000
adim = False

num_bursts = 2
skip_bursts = 2
simtime = 3*(num_bursts + skip_bursts)*di_param['tau_w']


'''
Theoretical computation.

Use the Fardet2017_SynchroBurst class to compute the theoretical time evolution
from the equivalent model.
'''

# theoretical values
theo = Fardet2017_SynchroBurst(di_param)
ts, Vs, ws, stimes = theo.time_evolution(
    model=model, homogeneous=True, num_bursts=num_bursts, steps=steps,
    adim=adim, show=False)

ts += 5 # shift time to align first bursts ends


'''
Simulation (requires NEST)

Compute the time evolution for the "aeif_psc_alpha" model in NEST.
Ignore the first `skip_bursts` network bursts to get rid of the transient.
'''

sim = Simulator_SynchroBurst(num_neurons, di_param, omp=10)
ts_simu, Vs_simu, ws_simu = sim.time_evolution(
    simtime, start_burst=skip_bursts, num_bursts=num_bursts, mbis=delay+20.,
    show=False)


'''
Plot the time evolution of the theoretical model
'''

# figure
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# Vs
ax1.plot(ts, Vs, c="k", lw=2.)
ax1.set_ylabel("V (mV)")
ax1.set_xlabel("Time (ms)")
# spikes
V_spike = theo.Vp / 2. if adim else (theo.Vt_dim + di_param["V_peak"]) / 2.
ax1.scatter(stimes, np.repeat(V_spike, len(stimes)), marker=".", c="k")
# ws
ax2.plot(ts, ws, c="k", lw=2.)
ax2.set_ylabel("w (pA)")


'''
Add the simulated curves
'''

ax1.plot(ts_simu, Vs_simu, c="r")
ax2.plot(ts_simu, ws_simu, c="b")

ax1.set_xlim([-0.01*ts_simu[-1], 1.01*ts_simu[-1]])


'''
Add the inset to zoom on the burst
'''

# zoom on the spikes
axins = inset_axes(ax1, width="30%", height="40%", loc=7)
axins.plot(ts_simu, Vs_simu, c="r")
axins.scatter(stimes, np.repeat(V_spike, len(stimes)), marker=".", c="k")
# sub region of the original image
y1, y2 = 1.1*di_param["V_reset"], 0.85*V_spike
xorigin = sim.phases["bursting"][num_bursts - 1][1] if num_bursts > 0 else 0
x1, x2 = np.subtract(sim.phases["bursting"][num_bursts], xorigin)
x1 = min(stimes[0], x1)
axins.axis([0.98*x1, 1.03*x2, y1, y2])
mark_inset(ax1, axins, loc1=3, loc2=1, fc="none", ec="0.5")
axins.tick_params(
    which="both", bottom="off", left="off", labelbottom="off", labelleft="off")

ax1.set_ylim(1.9*Vs_simu.min(), 0.8*V_spike)
ax2.set_ylim(0.9*ws_simu.min(), 2*ws_simu.max())


''' Show '''

plt.tight_layout()
plt.show()
