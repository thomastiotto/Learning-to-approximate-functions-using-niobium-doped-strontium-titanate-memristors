import nengo
from memristor_learning.MemristorHelpers import *
from memristor_learning.MemristorControllers import MemristorArray
from memristor_learning.MemristorModels import MemristorAnoukPair
from memristor_learning.MemristorLearningRules import mHopfieldHebbian

# hyperparameters
neurons = 3
dimensionality = 3
simulation_time = 10.0
learning_time = 5.0
simulation_step = 0.001

with nengo.Network() as model:
    inp = nengo.Node(
            lambda t, x: [ 1, 1, 0 ] if t < learning_time else np.random.randint( 2, size=neurons ),
            size_in=1
            )
    learning_switch = nengo.Node(
            lambda t, x: 1 if t < learning_time else 0,
            size_in=1
            )
    pre = nengo.Ensemble(
            n_neurons=neurons,
            dimensions=neurons,
            encoders=np.eye( neurons ),
            #  intercepts=nengo.dists.Choice([0.5]),
            intercepts=nengo.dists.Exponential( 0.15, 0.5, 1. ),
            label="Nodes"
            )
    memr_arr = MemristorArray(
            model=MemristorAnoukPair,
            learning_rule=mHopfieldHebbian(),
            in_size=neurons,
            out_size=neurons
            )
    learn = nengo.Node(
            output=memr_arr,
            size_in=neurons + 1,
            size_out=neurons,
            label="Learn"
            )
    
    gate_learn = nengo.Node(
            (lambda t, x: 0 if t < learning_time else x),
            size_in=neurons,
            size_out=neurons
            )
    
    nengo.Connection( inp, pre, synapse=None )
    nengo.Connection( pre.neurons, learn[ :neurons ], synapse=0.005 )
    
    nengo.Connection( learning_switch, learn[ -1 ], synapse=None )
    nengo.Connection( learn, gate_learn, synapse=None )
    nengo.Connection( gate_learn, pre.neurons, synapse=None )
    
    inp_probe = nengo.Probe( inp )
    pre_spikes_probe = nengo.Probe( pre.neurons )
    pre_probe = nengo.Probe( pre, synapse=0.01 )
    
    # nm.plot_network( model )

"""with nengo.Simulator( model, dt=simulation_step, optimize=True ) as sim:
    sim.run( simulation_time )

plot_ensemble_spikes( sim, "Pre", pre_spikes_probe, pre_probe )
# plot_ensemble_spikes( sim, "Post", post_spikes_probe, post_probe )
plot_ensemble( sim, pre_probe, time=learning_time )
if neurons <= 10:
    memr_arr.plot_state( sim, "conductance", combined=True )
    for t in range( 0, int( learning_time + 1 ), 1 ):
        memr_arr.plot_weight_matrix( time=t )

# plot_tuning_curves( pre, sim )

# print( "Mean squared error:", mse( sim, inp_probe, post_probe, learning_time, simulation_step ) )
print( f"Starting sparsity: {sparsity_measure( memr_arr.get_history( 'weight' )[ 0 ] )}" )
print( f"Ending sparsity: {sparsity_measure( memr_arr.get_history( 'weight' )[ -1 ] )}" )
print( "Pattern(s) learned" )
print( sim.data[ inp_probe ][ 0 ] )
print( "Pattern(s) for test" )
print( sim.data[ inp_probe ][ -1 ] )
print( "Final Ensemble Value" )
print( sim.data[ pre_probe ][ -1 ] )"""

memr_arr.memristors[ 0, 0 ].mem_plus.plot_memristor_curve()
