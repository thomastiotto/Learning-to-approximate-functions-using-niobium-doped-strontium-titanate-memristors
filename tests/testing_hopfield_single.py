import nengo
from nengo.networks import EnsembleArray
from memristor_learning.MemristorHelpers import *
from memristor_learning.MemristorControllers import MemristorArray
from memristor_learning.MemristorModels import MemristorAnoukPair
from memristor_learning.MemristorLearningRules import mHopfieldHebbian


def generate_encoders( n_neurons ):
    if n_neurons % 2 == 0:
        return [ [ -1 ] ] * int( (n_neurons / 2) ) + [ [ 1 ] ] * int( (n_neurons / 2) )
    else:
        return [ [ -1 ] ] * int( (n_neurons / 2) ) + [ [ 1 ] ] + [ [ 1 ] ] * int( (n_neurons / 2) )


# TODO add function_to_learn
# hyperparameters
neurons = 10
input = [ 0, 1 ]
dimensionality = len( input )
simulation_time = 30.0
learning_time = 15.0
simulation_step = 0.001
pre_nrn = neurons
post_nrn = neurons

with nengo.Network() as model:
    inp = nengo.Node( input )
    pre = nengo.Ensemble(
            n_neurons=pre_nrn,
            dimensions=dimensionality,
            label="Pre"
            )
    post = nengo.Ensemble(
            n_neurons=post_nrn,
            dimensions=dimensionality,
            label="Post"
            )
    
    memr_arr = MemristorArray(
            model=MemristorAnoukPair,
            learning_rule=mHopfieldHebbian(),
            in_size=pre_nrn,
            out_size=post_nrn,
            dimensions=[ pre.dimensions, post.dimensions ]
            )
    learn = nengo.Node(
            output=memr_arr,
            size_in=pre_nrn + post_nrn,
            size_out=post_nrn,
            label="Learn"
            )
    
    nengo.Connection( inp, pre, synapse=None )
    
    nengo.Connection( pre.neurons, learn[ :pre_nrn ], synapse=0.005 )
    # nengo.Connection( learn, post, synapse=None )
    # nengo.Connection( post.output, learn[ pre_nrn:pre_nrn + post_nrn ], synapse=None )
    
    inp_probe = nengo.Probe( inp )
    pre_spikes_probe = nengo.Probe( pre.neurons )
    # post_spikes_probe = nengo.Probe( post.neuron_output )
    pre_probe = nengo.Probe( pre, synapse=0.01 )
    # post_probe = nengo.Probe( post, synapse=0.01 )
    
    # nm.plot_network( model )

with nengo.Simulator( model, dt=simulation_step, optimize=True ) as sim:
    sim.run( simulation_time )

plot_ensemble_spikes( sim, "Pre", pre_spikes_probe, pre_probe )
# plot_ensemble_spikes( sim, "Post", post_spikes_probe, post_probe )
# plot_pre_post( sim, pre_probe, post_probe, inp_probe, time=learning_time )
# if neurons <= 10:
#     memr_arr.plot_state( sim, "conductance", combined=True )
#     for t in range( 0, int( learning_time + 1 ), 1 ):
#         memr_arr.plot_weight_matrix( time=t )
#
# print( "Mean squared error:", mse( sim, inp_probe, post_probe, learning_time, simulation_step ) )
# print( f"Starting sparsity: {sparsity_measure( memr_arr.get_history( 'weight' )[ 0 ] )}" )
# print( f"Ending sparsity: {sparsity_measure( memr_arr.get_history( 'weight' )[ -1 ] )}" )
