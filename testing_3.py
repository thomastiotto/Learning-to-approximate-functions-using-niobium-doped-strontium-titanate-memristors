from neuromorphic_library_3 import MemristorArray
import neuromorphic_library_3 as nm
import nengo
import numpy as np

# import nengo_ocl

# hyperparameters
input_period = 4.0
input_frequency = 1 / input_period
pre_nrn = 4
post_nrn = 4
type = "pair"
spike_learning = True
function_to_learn = lambda x: x

with nengo.Network() as model:
    inp = nengo.Node(
            output=lambda t: np.sin( input_frequency * 2 * np.pi * t ),
            size_out=1,
            label="Input"
    )
    # TODO use Izichevich model instead of LIF?
    pre = nengo.Ensemble( pre_nrn,
                          dimensions=1,
                          encoders=[ [ -1 ] ] * int( (pre_nrn / 2) ) + [ [ 1 ] ] * int( (pre_nrn / 2) ),
                          label="Pre" )
    post = nengo.Ensemble( post_nrn,
                           dimensions=1,
                           encoders=[ [ -1 ] ] * int( (post_nrn / 2) ) + [ [ 1 ] ] * int( (post_nrn / 2) ),
                           label="Post" )
    err = nengo.Ensemble( 100,
                          dimensions=1,
                          radius=2,
                          label="Err" )
    
    memr_arr = MemristorArray( pre_nrn, post_nrn, type=type, spike_learning=spike_learning )
    learn = nengo.Node( memr_arr, size_in=pre_nrn + err.dimensions, size_out=post_nrn, label="Learn" )
    
    nengo.Connection( inp, pre )
    
    nengo.Connection( pre.neurons, learn[ :pre_nrn ], synapse=0.01 )
    nengo.Connection( err, learn[ pre_nrn: ], synapse=0.01 )
    nengo.Connection( learn, post.neurons, synapse=None )
    
    nengo.Connection( pre, err, function=function_to_learn, transform=-1 )
    nengo.Connection( post, err )
    
    inp_probe = nengo.Probe( inp )
    pre_spikes_probe = nengo.Probe( pre.neurons )
    post_spikes_probe = nengo.Probe( post.neurons )
    pre_probe = nengo.Probe( pre, synapse=0.01 )
    post_probe = nengo.Probe( post, synapse=0.01 )
    err_probe = nengo.Probe( err, synapse=0.01 )
    
    
    def inhibit( t ):
        return 2.0 if t > 20.0 else 0.0
    
    
    inhib = nengo.Node( inhibit )
    nengo.Connection( inhib, err.neurons, transform=[ [ -1 ] ] * err.n_neurons )
    
    pre_filt_probe = nengo.Probe( pre.neurons, synapse=0.01 )

sim_time = 30
with nengo.Simulator( model ) as sim:
    sim.run( sim_time )

# nm.plot_network( model )
nm.plot_ensemble_spikes( sim, "Pre", pre_spikes_probe, inp_probe )
nm.plot_ensemble_spikes( sim, "Post", post_spikes_probe, pre_probe )
nm.plot_pre_post( sim, pre_probe, post_probe, inp_probe, err_probe )
memr_arr.plot_state( sim, "conductance", combined=True )
memr_arr.plot_state( sim, "conductance", combined=False )
