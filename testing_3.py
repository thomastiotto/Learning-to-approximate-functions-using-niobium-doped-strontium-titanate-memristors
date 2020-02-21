from neuromorphic_library_3 import MemristorArray
from neuromorphic_library_3 import plot_network
import nengo
import numpy as np

# hyperparameters
input_period = 4.0
input_frequency = 1 / input_period
pre_nrn = 100
post_nrn = 100

with nengo.Network() as model:
    inp = nengo.Node(
            output=lambda t: np.sin( input_frequency * 2 * np.pi * t ),
            size_out=1,
            label="Input"
    )
    # TODO use Izichevich model instead of LIF?
    pre = nengo.Ensemble( pre_nrn, dimensions=1, label="Pre" )
    post = nengo.Ensemble( post_nrn, dimensions=1, label="Post" )
    err = nengo.Ensemble( 100, dimensions=1, label="Err" )
    
    memr_arr = MemristorArray( pre_nrn, post_nrn, "pair" )
    learn = nengo.Node( memr_arr, size_in=pre_nrn + err.dimensions, size_out=post_nrn, label="Learn" )
    
    nengo.Connection( inp, pre )
    
    nengo.Connection( pre.neurons, learn[ :pre_nrn ], synapse=0.01 )
    nengo.Connection( err, learn[ pre_nrn: ], synapse=0.01 )
    nengo.Connection( learn, post.neurons, synapse=None )
    
    nengo.Connection( pre, err, function=lambda x: x, transform=-1 )
    nengo.Connection( post, err )
    
    # plot_network(model)
    
    inp_probe = nengo.Probe( inp )
    pre_spikes_probe = nengo.Probe( pre.neurons )
    pre_filt_probe = nengo.Probe( pre.neurons, synapse=0.01 )
    pre_probe = nengo.Probe( pre, synapse=0.01 )
    post_probe = nengo.Probe( post, synapse=0.01 )
    err_probe = nengo.Probe( err, synapse=0.01 )

sim_time = 30
with nengo.Simulator( model, dt=0.01 ) as sim:
    sim.run( sim_time )

import datetime
from nengo.utils.matplotlib import rasterplot
import matplotlib.pyplot as plt

# plot spikes from pre
plt.figure()
# plt.suptitle( datetime.datetime.now().strftime( '%H:%M:%S %d-%m-%Y' ) )
fig, ax1 = plt.subplots()
ax1 = plt.subplot( 1, 1, 1 )
rasterplot( sim.trange(), sim.data[ pre_spikes_probe ], ax1 )
ax1.set_xlim( 0, sim_time )
ax1.set_ylabel( 'Neuron' )
ax1.set_xlabel( 'Time (s)' )
ax2 = plt.twinx()
ax2.plot( sim.trange(), sim.data[ inp_probe ], c="k" )
plt.title( "Neural activity" )
plt.show()

# plot input, neural representations and error
plt.figure()
# plt.suptitle( datetime.datetime.now().strftime( '%H:%M:%S %d-%m-%Y' ) )
plt.subplot( 2, 1, 1 )
plt.plot( sim.trange(), sim.data[ inp_probe ], c="k", label="Input" )
plt.plot( sim.trange(), sim.data[ pre_probe ], c="b", label="Pre" )
plt.plot( sim.trange(), sim.data[ post_probe ], c="g", label="Post" )
plt.title( "Values" )
plt.legend( loc='best' )
plt.subplot( 2, 1, 2 )
plt.plot( sim.trange(), sim.data[ err_probe ], c="r" )
plt.title( "Error" )
plt.show()

memr_arr.plot_state( sim, "conductance" )
