#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot the attractor for different networks """

import nngt
import nest

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.mlab import griddata
from matplotlib.markers import MarkerStyle
plt.rcParams['axes.grid'] = False

import numpy as np
from scipy.signal import argrelmax, argrelmin

from PyNeurActiv.models import Fardet2017_SynchroBurst, Simulator_SynchroBurst
from tools_burst_cycles import get_data


'''
Parameters
----------

* for the neuronal properties
* for the network properties
* for the simulation
'''

from parameters import (di_param, num_neurons, w_prop, avg_deg, std_deg1,
                        std_deg2, delay)

gL = di_param["g_L"]
EL = di_param["E_L"]
a = di_param["a"]
DT = di_param["Delta_T"]
Vth = di_param["V_th"]
Ie = di_param["I_e"]
tw = di_param["tau_w"]
V_spike = -35.            # arbitrary value for potential at spike time

# SIMULATION
chosen_interval = [3, 7]  # bursts between which we do the recording
omp = 10                  # number of OpenMP threads
simtime = chosen_interval[1]*(3*di_param["tau_w"]) # duration of the simulation
resolution = 0.1          # time resolution for the simulation
num_avg = 2               # number of realizations to compute average and std

# PLOT
V_samples = 200           # number of samples for the histogram along V
w_samples = 200           # number of samples for the histogram along w


'''
Tasks
-----

Define the networks to simulate and for which the attractor will be plotted.
Generate the population of neurons the will be created from.
'''

networks = {}
networks["gid1"] = {'graph_type': 'gaussian_degree', 'avg': avg_deg,
   'std': std_deg1}
networks["fid"] = {'graph_type': 'fixed_degree', 'degree': avg_deg}
networks["gid2"] = {'graph_type': 'gaussian_degree', 'avg': avg_deg,
   'std': std_deg2}

# population of neurons
pop = nngt.NeuralPop.uniform(num_neurons, neuron_model="aeif_psc_alpha",
                             neuron_param=di_param)

'''
Density plots
-------------

Functions taking care of the density plots for the Gaussian networks.
'''


def rgb_to_hex(rgb):
    rgb = tuple(np.multiply(255, rgb).astype(int))
    if len(rgb) == 3:
        return '#%02x%02x%02x' % rgb
    elif len(rgb) == 4:
        return '#%02x%02x%02x' % rgb[:3]
    else:
        raise ArgumentError("Invalid rgb(a) array")


def density_field(lst_Vs, lst_ws):
    '''
    Get the density of the trajectory in phase space.

    Parameters
    ----------
    lst_Vs : list of np.arrays
        List of the potentials for each series of runs.
    lst_ws :  : list of np.arrays
        List of the adaptation variables for each series of runs.

    Returns
    -------
    Vi, wi, zi : np.arrays
        Gridded interpolated results for the potential, the adaptation variable
        and the density of states.
    '''
    # get histogram
    x, y = [], []
    for Vs, ws in zip(lst_Vs, lst_ws):
        x.extend(Vs)
        y.extend(ws)
    Vmin, Vmax = 1.01*np.min(x), 0.99*np.max(x)
    wmin, wmax = 0.99*np.min(y), 1.01*np.max(y)
    arr = np.array(x)
    xedges = np.linspace(Vmin, Vmax, V_samples)
    yedges = np.linspace(wmin, wmax, w_samples)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    xedges = xedges[:-1] + 0.5*np.diff(xedges)
    yedges = yedges[:-1] + 0.5*np.diff(yedges)
    x = np.repeat(xedges, len(yedges))
    y = np.tile(yedges, len(xedges))
    z = np.reshape(H, len(yedges)*len(xedges))
    arr = np.array((x,y))
    # define grid and grid the data
    Vi = np.linspace(Vmin, Vmax, 3*V_samples)
    wi = np.linspace(wmin, wmax, 3*w_samples)
    zi = griddata(x, y, z, Vi, wi, interp='linear')
    #~ zi = spgrid((x, y), z, (Vi, wi), method='linear')
    zi[zi <= 0] = 1e-1
    zi = np.log(zi)
    return Vi, wi, zi


'''
Run the simulations and average on the fly.

Only once for the fixed in-degree (not fluctuations), but average the runs
for the Gaussian in-degree networks.
'''

if __name__ == "__main__":
    di_res = {}
    V_min, w_min, V_max, w_max = np.inf, np.inf, -np.inf, -np.inf
    burst_start, burst_end = chosen_interval

    cvals = {"fid": 0.9, "gid1": 0.45, "gid2": 0.05}
    cmaps = {}

    for graph, instructions in networks.items():
        cval = cvals[graph]
        if graph == "fid":
            # generate the graph
            net = nngt.generate(instructions, population=pop, weights=w_prop)
            # run the simulation and record the mean-field behaviour
            testSim = Simulator_SynchroBurst.from_nngt_network(
                net, resolution=resolution, omp=omp)
            resultsSim = testSim.compute_properties(
                simtime=simtime, steady_state=burst_start)
            Vs, ws = get_data(
                testSim, burst_start, burst_start + 1, num_neurons, Vth)
            # get minimal values for
            V_min = min(V_min, Vs.min())
            w_min = min(w_min, ws.min())
            V_max = max(V_max, Vs.max())
            w_max = max(w_max, ws.max())
            di_res["fid"] = (Vs, ws)
            cmaps[graph] = cval
        else:
            lst_Vs, lst_ws = [], []
            for i in range(num_avg):
                # generate the graph
                net = nngt.generate(
                    instructions, population=pop, weights=w_prop)
                # run the simulation and record the mean-field behaviour
                testSim = Simulator_SynchroBurst.from_nngt_network(
                    net, omp=omp)
                resultsSim = testSim.compute_properties(
                    simtime=simtime, steady_state=burst_start)
                Vs, ws = get_data(
                    testSim, burst_start, burst_end, num_neurons, Vth)
                lst_Vs.append(Vs)
                lst_ws.append(ws)
            # get the color
            r, g, b, _ = cm.viridis(cval)
            cdict = {'red':   [(0.0, r, r), (1.0, r, r)],
                     'green': [(0.0, g, g), (1.0, g, g)],
                     'blue':  [(0.0, b, b), (1.0, b, b)]}
            # Create cmap with alpha
            cmap = LinearSegmentedColormap('map_{}'.format(graph), cdict, 256)
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
            cmaps[graph] = ListedColormap(my_cmap)
            # get the density
            Vs_grid, ws_grid, density = density_field(lst_Vs, lst_ws)
            V_min = min(V_min, Vs_grid.min())
            w_min = min(w_min, ws_grid.min())
            V_max = max(V_max, Vs_grid.max())
            w_max = max(w_max, ws_grid.max())
            di_res[graph] = (Vs_grid, ws_grid, density)


    '''
    Plot the results:

    * one fixed in-degree network
    * two Gaussian networks with <k> = 100 and std_k = {5, 30}
    * the continuous model
    * the nullcline
    * the recovery path
    '''

    fig, ax = plt.subplots()

    for graph, data in di_res.items():
        if graph == "fid":
            Vs, ws = data
            cval = cmaps[graph]
            # set spike times
            idx_spiking = np.where(Vs > (Vth + 4*DT))[0]
            idx_spikes = argrelmax(Vs)[0]+1
            idx_spikes = [idx for idx in idx_spikes
                          if Vs[idx-1] - Vs[idx] > DT]
            Vs_tmp, ws_tmp = Vs.copy(), ws.copy()
            Vs, ws = Vs.tolist(), ws.tolist()
            for i,idx in enumerate(idx_spikes):
                Vs.insert(idx+i, np.NaN)
                ws.insert(idx+i, np.NaN)
                Vs[idx+i-1] = V_spike
                ax.scatter(
                    Vs[idx+i-1], ws[idx+i-1], marker=MarkerStyle('s', 'none'),
                    edgecolor=cm.viridis(cmaps[graph]))
                ax.scatter(
                    Vs[idx+i+1], ws[idx+i+1], edgecolor=cm.viridis(cval),
                    marker=MarkerStyle('o', 'none'))
            # plot the trajectory in phase space
            ax.plot(Vs, ws, c=cm.viridis(cval))
        else:
            Vs, ws, density = data
            ax.contourf(
                Vs, ws, density, 10, cmap=cmaps[graph], vmin=0)
            ls = "dashdot" if graph == "gid1" else "dotted"
            c = rgb_to_hex(cmaps[graph](1))
            ax.contour(Vs, ws, density, levels=[0], colors=c, linestyles=ls)

    # plot the nullcline
    Vs = np.linspace(1.05*V_min, V_max, 1000) # V is negative
    ws = -gL*((Vs - EL) - DT*np.exp((Vs-Vth)/DT)) + Ie
    ax.plot(Vs, ws, c="k", linestyle="--")

    # plot the \dot{w} = -\dot{V} line
    di_param["model"] = "aeif_psc_alpha"
    di_param["delay"] = delay
    di_param["avg_deg"] = avg_deg
    di_param["weight"] = w_prop["value"]
    theo = Fardet2017_SynchroBurst(di_param)
    Vs2 = np.linspace(1.05*V_min, 0.95*Vth, 1000)
    ws2 = theo.w_recover_lin(Vs2)
    ax.plot(Vs2, ws2, c="k", linestyle=":")

    ax.set_ylim([0.95*w_min, 1.05*w_max])
    ax.set_xlim([1.05*V_min, 0.95*V_max])
    ax.set_xlabel('V (mV)')
    ax.set_ylabel('w (pA)')
    plt.legend(loc=3)

    plt.show()
