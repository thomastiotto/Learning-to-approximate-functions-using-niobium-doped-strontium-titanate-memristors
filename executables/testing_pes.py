import nengo
from memristor_learning.MemristorHelpers import *
from memristor_learning.MemristorControllers import MemristorArray
from memristor_learning.MemristorModels import MemristorAnoukPair, MemristorAnoukBidirectional
from memristor_learning.MemristorLearningRules import mPES

# hyperparameters
neurons = 4
simulation_time = 30.0
learning_time = 15.0
simulation_step = 0.001
function_to_learn = lambda x: x
input_period = 4.0
input_frequency = 1 / input_period
pre_nrn = neurons
post_nrn = neurons
err_nrn = neurons
seed = 0

with nengo.Network() as model:
    inp = nengo.Node(
            output=lambda t: np.sin( input_frequency * 2 * np.pi * t ),
            size_out=1,
            label="Input"
            )
    learning_switch = nengo.Node(
            lambda t, x: 1 if t < learning_time else 0,
            size_in=1
            )
    
    
    def generate_encoders( n_neurons ):
        if n_neurons % 2 == 0:
            return [ [ -1 ] ] * int( (n_neurons / 2) ) + [ [ 1 ] ] * int( (n_neurons / 2) )
        else:
            return [ [ -1 ] ] * int( (n_neurons / 2) ) + [ [ 1 ] ] + [ [ 1 ] ] * int( (n_neurons / 2) )
    
    
    pre = nengo.Ensemble(
            n_neurons=pre_nrn,
            dimensions=1,
            encoders=generate_encoders( pre_nrn ),
            label="Pre",
            seed=seed
            )
    post = nengo.Ensemble(
            n_neurons=post_nrn,
            dimensions=1,
            encoders=generate_encoders( post_nrn ),
            label="Post",
            seed=seed
            )
    error = nengo.Ensemble(
            n_neurons=err_nrn,
            dimensions=1,
            label="Error",
            seed=seed
            )
    
    # TODO get encoders at runtime as sim.data[ens].encoders
    memr_arr = MemristorArray(
            model=MemristorAnoukPair,
            learning_rule=mPES( post.encoders ),
            in_size=pre_nrn,
            out_size=post_nrn,
            seed=seed
            )
    learn = nengo.Node(
            output=memr_arr,
            size_in=pre_nrn + error.dimensions + 1,
            size_out=post_nrn,
            label="Learn"
            )
    
    nengo.Connection( inp, pre )
    nengo.Connection( pre.neurons, learn[ :pre_nrn ], synapse=0.005 )
    nengo.Connection( post, error )
    nengo.Connection( pre, error, function=function_to_learn, transform=-1 )
    nengo.Connection( error, learn[ pre_nrn:-1 ] )
    nengo.Connection( learn, post.neurons, synapse=None )
    nengo.Connection( learning_switch, learn[ -1 ], synapse=None )
    
    inp_probe = nengo.Probe( inp )
    pre_spikes_probe = nengo.Probe( pre.neurons )
    post_spikes_probe = nengo.Probe( post.neurons )
    pre_probe = nengo.Probe( pre, synapse=0.01 )
    post_probe = nengo.Probe( post, synapse=0.01 )
    
    # plot_network( model )

with nengo.Simulator( model, dt=simulation_step ) as sim:
    sim.run( simulation_time )

plot_ensemble_spikes( sim, "Pre", pre_spikes_probe, pre_probe )
plot_ensemble_spikes( sim, "Post", post_spikes_probe, post_probe )
plot_pre_post( sim, pre_probe, post_probe, inp_probe, memr_arr.get_history( "error" ), time=learning_time )
if neurons <= 10:
    memr_arr.plot_state( sim, "conductance", combined=True )
    for t in range( 0, int( learning_time + 1 ), 1 ):
        memr_arr.plot_weight_matrix( time=t )

print( "Mean squared error:", mse( sim, inp_probe, post_probe, learning_time, simulation_step ) )
print( f"Starting sparsity: {sparsity_measure( memr_arr.get_history( 'weight' )[ 0 ] )}" )
print( f"Ending sparsity: {sparsity_measure( memr_arr.get_history( 'weight' )[ -1 ] )}" )
