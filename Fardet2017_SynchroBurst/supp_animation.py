"""
======================
Supplemetary animation
======================

Subclass nngt.plot.Animation2d to get the vector field in phase space.
"""

import numpy as np

import nest
nest.SetKernelStatus({"local_num_threads": 10})
resol = 0.1
nest.SetKernelStatus({"resolution": resol})

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nngt
from nngt.simulation import randomize_neural_states, monitor_nodes
from nngt.plot import Animation2d


# ---------- #
# Parameters #
# ---------- #

# matplotlib
matplotlib.rc('font', family='normal', size='14')
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10)

# nest
simtime = 1000.      # duration of the simulation
di_param = {         # neuronal parameters
    'V_reset': -55.,
    'V_peak': 0.0,
    'V_th': -50.,
    'I_e': 400.,
    'g_L': 11.7,
    'tau_w': 900.,
    'E_L': -65.,
    'Delta_T': 2.,
    'a': 2.8,
    'b': 36.3,
    'C_m': 200.,
    'V_m': -70.,
    'w': 100.,
    'tau_syn_ex': 2.
}

# network
num_neurons = 1000   # number of neurons in the graph
avg_degree = 100     # average number of neighbours
std_degree = 5       # deviation for the Gaussian graph
weight = 5.          # synaptic weight
delay = 20.          # propagation delay of the spikes


# ------- #
# Network #
# ------- #

# population of neurons
pop = nngt.NeuralPop.uniform(
    num_neurons, neuron_model="aeif_psc_alpha", neuron_param=di_param)

# fixed in-degree
net = nngt.generation.fixed_degree(
    avg_degree, population=pop, weights=weight, delays=delay)


# ---------- #
# NEST stuff #
# ---------- #

gids = net.to_nest()

mm_param = {
    'record_from': ['V_m', 'w', "I_syn_ex"],
    'to_accumulator': True,
    'interval': resol
}

(mm, sd), recordables = monitor_nodes(
    gids, nest_recorder=['multimeter', 'spike_detector'],
    params=[mm_param, {}])

randomization = {
    #~ "V_m": ("uniform", -70., -50.),
    #~ "w": ("uniform", 20., 300.),
    #~ "I_e": ("normal", di_param["I_e"], 10.)
}

randomize_neural_states(net, randomization)

nest.Simulate(simtime)

# --------- #
# Animation #
# --------- #

def dotV(V, w, time_dependent):
    gL, DT = di_param['g_L'], di_param['Delta_T']
    Ie, Cm = di_param['I_e'], di_param['C_m']
    leak = -gL*(V - di_param['E_L'])
    spike = gL*DT*np.exp((V - di_param['V_th']) / DT)
    speed = (leak + spike - w + Ie + np.sum(time_dependent)) / Cm
    return np.sign(speed) * np.sqrt(np.abs(speed))


def dotw(V, w, time_dependent):
    speed = (di_param['a']*(V - di_param['E_L']) - w) / di_param['tau_w']
    return np.sign(speed) * np.sqrt(np.abs(speed))


def w_Vnull(Vs, I):
    gL = di_param["g_L"]
    EL = di_param["E_L"]
    VT = di_param["V_th"]
    DT = di_param["Delta_T"]
    Ie = di_param["I_e"]
    return -gL * ((Vs-EL) - DT*np.exp((Vs-VT) / DT)) + Ie + I


class AnimNullcline(Animation2d):
    
    def __init__(self, spike_detector, multimeter, start=0., timewindow=None,
                 trace=5., sort_neurons=False, interval=10, network=None,
                 **kwargs):
        ''' Init, then add nullcline '''
        super(AnimNullcline, self).__init__(spike_detector, multimeter,
            start=start, timewindow=timewindow, trace=trace, x='V_m', y='w',
            sort_neurons=sort_neurons, network=network, make_rate=False,
            **kwargs)
        # nullcline
        self.line_Vnull = Line2D([], [], color='blue')
        V_min, V_max = self.ps.get_xlim()
        self.Vs = np.linspace(V_min, V_max, 200)
        self.ps.add_line(self.line_Vnull)
        # currents
        Is = nest.GetStatus(multimeter)[0]["events"]["I_syn_ex"]
        self.Is = Is[self.idx_start:] / self.num_neurons
        self.set_axis(
            self.second, xlabel='Time (ms)', ylabel='$I_{syn, ex}$ (pA)',
            lines=self.lines_second, xdata=self.times, ydata=self.Is)
        # replace draw function
        self._func = self._draw

    def _draw(self, framedata):
        i = framedata
        head = i - 1
        head_slice = ((self.times > self.times[i] - self.trace)
                      & (self.times < self.times[i]))
        # draw nullcline
        ws = w_Vnull(self.Vs, self.Is[i])
        self.line_Vnull.set_data(self.Vs, ws)
        # get and return lines
        
        lines = super(AnimNullcline, self)._draw(framedata)
        # replace rate by synaptic current
        self.line_second_.set_data(self.times[:i], self.Is[:i])
        self.line_second_a.set_data(self.times[head_slice], self.Is[head_slice])
        self.line_second_e.set_data(self.times[head], self.Is[head])
        lines.append(self.line_Vnull)
        return lines
    
    def _init_draw(self):
        super(AnimNullcline, self)._init_draw()
        self.line_Vnull.set_data([], [])


# ---- #
# Show #
# ---- #

ani = AnimNullcline(
    sd, mm, start=500., trace=10., timewindow=500.,
    interval=1, sort_neurons="in-degree", network=net,
    vector_field=True, dotx=dotV, doty=dotw, time_dependent=['I_syn_ex'],
    figsize=(14, 8), dpi=150)

#~ ani.save_movie('nullcline_fid.mp4', fps=16, interval=10)
plt.show()
