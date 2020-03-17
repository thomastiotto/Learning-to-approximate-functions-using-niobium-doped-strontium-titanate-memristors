from neuromorphic_library import MemristorArray
import neuromorphic_library as nm
import nengo
import numpy as np
import nengo_ocl

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
err_nrn = neurons

with nengo.Network() as model:
    inp = nengo.Node(
            output=lambda t: np.sin( input_frequency * 2 * np.pi * t ),
            size_out=1,
            label="Input"
            )
    
    
    def generate_encoders( n_neurons ):
        if n_neurons % 2 == 0:
            return [ [ -1 ] ] * int( (n_neurons / 2) ) + [ [ 1 ] ] * int( (n_neurons / 2) )
        else:
            return [ [ -1 ] ] * int( (n_neurons / 2) ) + [ [ 1 ] ] + [ [ 1 ] ] * int( (n_neurons / 2) )
    
    
    # TODO get encoders at runtime
    pre = nengo.Ensemble( n_neurons=pre_nrn,
                          dimensions=1,
                          encoders=generate_encoders( pre_nrn ),
                          # intercepts=[ 0.1 ] * pre_nrn,
                          label="Pre" )
    post = nengo.Ensemble( n_neurons=post_nrn,
                           dimensions=1,
                           encoders=generate_encoders( post_nrn ),
                           # intercepts=[ 0.1 ] * pre_nrn,
                           label="Post" )
    error = nengo.Ensemble( n_neurons=err_nrn,
                            dimensions=1,
                            label="Error" )
    
    memr_arr = MemristorArray( learning_rule="mPES",
                               post_encoders=post.encoders,  # sim.data[ens].encoders
                               in_size=pre_nrn,
                               out_size=post_nrn,
                               dimensions=[ pre.dimensions, post.dimensions ],
                               logging=True
                               )
    learn = nengo.Node( output=memr_arr,
                        size_in=pre_nrn + error.dimensions,
                        size_out=post_nrn,
                        label="Learn" )
    
    nengo.Connection( inp, pre )
    nengo.Connection( pre.neurons, learn[ :pre_nrn ], synapse=0.005 )
    nengo.Connection( post, error )
    nengo.Connection( pre, error, function=function_to_learn, transform=-1 )
    nengo.Connection( error, learn[ pre_nrn: ] )
    nengo.Connection( learn, post.neurons, synapse=None )
    
    inp_probe = nengo.Probe( inp )
    pre_spikes_probe = nengo.Probe( pre.neurons )
    post_spikes_probe = nengo.Probe( post.neurons )
    pre_probe = nengo.Probe( pre, synapse=0.01 )
    post_probe = nengo.Probe( post, synapse=0.01 )
    
    
    # nm.plot_network( model )
    
    def inhibit( t ):
        return 2.0 if t > learning_time else 0.0
    
    
    inhib = nengo.Node( inhibit )
    nengo.Connection( inhib, error.neurons, transform=[ [ -1 ] ] * error.n_neurons )

with nengo.Simulator( model, dt=simulation_step ) as sim:
    sim.run( simulation_time )

nm.plot_ensemble_spikes( sim, "Pre", pre_spikes_probe, pre_probe )
nm.plot_ensemble_spikes( sim, "Post", post_spikes_probe, post_probe )
if neurons < 10:
    nm.plot_pre_post( sim, pre_probe, post_probe, inp_probe, memr_arr.get_history( "error" ), time=learning_time )
# memr_arr.plot_state( sim, "conductance", combined=True )

print( "Mean squared error:", nm.mse( sim, inp_probe, post_probe, learning_time, simulation_step ) )
