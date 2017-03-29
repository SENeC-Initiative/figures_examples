#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Parameters """


'''
Parameters
----------

* for the neuronal properties
* for the network properties
* for the simulation
'''

# NEURON
di_param = {
    'V_reset': -58.,
    'V_peak': 0.0,
    'V_th': -50.,
    'I_e': 300.,
    'g_L': 9.,
    'tau_w': 300.,
    'E_L': -70.,
    'Delta_T': 2.,
    'a': 2.,
    'b': 60.,
    'C_m': 200.,
    'V_m': -70.,
    'w': 100.,
    'tau_syn_ex': 0.2
}

di_param_penn = {
    'V_reset': -58.,
    'V_peak': 0.0,
    'V_th': -50.,
    'I_e': 300.,
    'g_L': 12.,
    'tau_w': 2000.,
    'E_L': -70.,
    'Delta_T': 2.,
    'a': 2.,
    'b': 60.,
    'C_m': 200.,
    'V_m': -70.,
    'w': 100.,
    'tau_syn_ex': 0.7
}

di_param_segal = {
    'V_reset': -48.,
    'V_peak': 0.0,
    'V_th': -43.,
    'I_e': 200.,
    'g_L': 9.,
    'tau_w': 5000.,
    'E_L': -60.,
    'Delta_T': 2.,
    'a': 2.,
    'b': 2.,
    'C_m': 250.,
    'V_m': -60.,
    'w': 100.,
    'tau_syn_ex': .9
}

# NETWORK
num_neurons = 1000
w_prop = { "distribution": "constant", "value": 60. } # synaptic strengths
avg_deg = 100             # average number of neighbours
std_deg1 = 5              # deviation for the 1st Gaussian graph
std_deg2 = 20             # deviation for the 2nd Gaussian graph

# SIMULATION
chosen_interval = [3, 7]  # bursts between which we do the recording
delay = 1.                # transmission delay for spikes
