from neuromorphic_library import MemristorArray
import neuromorphic_library as nm
import nengo
import numpy as np
import nengo_ocl


def generate_encoders( n_neurons ):
    if n_neurons % 2 == 0:
        return [ [ -1 ] ] * int( (n_neurons / 2) ) + [ [ 1 ] ] * int( (n_neurons / 2) )
    else:
        return [ [ -1 ] ] * int( (n_neurons / 2) ) + [ [ 1 ] ] + [ [ 1 ] ] * int( (n_neurons / 2) )


# hyperparameters
neurons = 4
simulation_time = 30.0
learning_time = 10.0
simulation_step = 0.001
function_to_learn = lambda x: x
input_period = 4.0
input_frequency = 1 / input_period
pre_nrn = neurons
post_nrn = neurons

with nengo.Network() as model:
    inp = nengo.Node( lambda t: np.sin( input_frequency * 2 * np.pi * t ) )
    pre = nengo.Ensemble(
            n_neurons=pre_nrn,
            dimensions=1,
            encoders=generate_encoders( pre_nrn ),
            # intercepts=[ 0.1 ] * pre_nrn,
            label="Pre"
            )
    post = nengo.Ensemble(
            n_neurons=post_nrn,
            dimensions=1,
            encoders=generate_encoders( post_nrn ),
            # intercepts=[ 0.1 ] * post_nrn,
            label="Post"
            )
    memr_arr = MemristorArray(
            learning_rule="mBCM",
            in_size=pre_nrn,
            out_size=post_nrn,
            dimensions=[ pre.dimensions, post.dimensions ],
            logging=True
            )
    learn = nengo.Node(
            output=memr_arr,
            size_in=pre_nrn + post_nrn * 2,
            size_out=post_nrn,
            label="Learn"
            )
    
    gate_inp_to_post = nengo.Node(
            (lambda t, x: x if t < learning_time else 0),
            size_in=1
            )
    gate_learn_to_post = nengo.Node(
            (lambda t, x: 0 if t < learning_time else x),
            size_in=post_nrn,
            size_out=post_nrn
            )
    gate_learn_from_post = nengo.Node(
            (lambda t, x: x if t < learning_time else 0),
            size_in=post_nrn * 2,
            size_out=post_nrn * 2
            )
    
    nengo.Connection( inp, pre, synapse=None )
    nengo.Connection( inp, gate_inp_to_post, synapse=None )
    nengo.Connection( gate_inp_to_post, post, synapse=None )
    
    nengo.Connection( pre.neurons, learn[ :pre_nrn ], synapse=0.005 )
    nengo.Connection( learn, gate_learn_to_post, synapse=None )
    nengo.Connection( gate_learn_to_post, post.neurons, synapse=None )
    
    nengo.Connection( post.neurons, gate_learn_from_post[ post_nrn: ], synapse=0.005 )
    nengo.Connection( gate_learn_from_post[ post_nrn: ], learn[ pre_nrn:pre_nrn + post_nrn ], synapse=None )
    theta_filter = nengo.Lowpass( tau=1.0 )
    nengo.Connection( post.neurons, gate_learn_from_post[ :post_nrn ], synapse=theta_filter )
    nengo.Connection( gate_learn_from_post[ :post_nrn ], learn[ pre_nrn + post_nrn: ], synapse=None )
    
    inp_probe = nengo.Probe( inp )
    pre_spikes_probe = nengo.Probe( pre.neurons )
    post_spikes_probe = nengo.Probe( post.neurons )
    pre_probe = nengo.Probe( pre, synapse=0.01 )
    post_probe = nengo.Probe( post, synapse=0.01 )
    
    # nm.plot_network( model )

with nengo.Simulator( model, dt=simulation_step, optimize=True ) as sim:
    sim.run( simulation_time )

nm.plot_ensemble_spikes( sim, "Pre", pre_spikes_probe, pre_probe )
nm.plot_ensemble_spikes( sim, "Post", post_spikes_probe, post_probe )
nm.plot_pre_post( sim, pre_probe, post_probe, inp_probe, time=learning_time )
if neurons < 10:
    memr_arr.plot_state( sim, "conductance", combined=True )
# for t in range( 0, int( learning_time + 1 ), 2 ):
#     memr_arr.plot_weight_matrix( time=t )

print( "Mean squared error:", nm.mse( sim, inp_probe, post_probe, learning_time, simulation_step ) )
