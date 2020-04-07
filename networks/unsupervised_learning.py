import argparse
from functools import partial
import nengo
from memristor_learning.MemristorHelpers import *
from memristor_learning.MemristorControllers import *
from memristor_learning.MemristorLearningRules import *
from memristor_learning.MemristorModels import *

parser = argparse.ArgumentParser( description="Test voltage converters" )
# parser.add_argument( "memristor_array", type=MemristorControllers, default=MemristorControllers.MemristorArray,
#                      help="the memristor controller model" )

parser.add_argument( "--neurons", metavar="N", type=int, default=4,
                     help="the number of neurons in the ensembles" )
parser.add_argument( "--simulation_time", metavar="T", type=int, default=30,
                     help="the total time to run the simulation for" )
parser.add_argument( "--learning_time", metavar="L", type=int, default=15,
                     help="the time learn for" )
parser.add_argument( "--simulation_step", metavar="s", type=float, default=0.001,
                     help="the Nengo stepsize for the simulation" )
parser.add_argument( "--function", metavar="F", default="lambda x: x",
                     help="the function to learn" )
parser.add_argument( "--input_period", metavar="p", type=float, default=4.0,
                     help="the period of the input sine wave" )
parser.add_argument( "--seed", metavar="s", type=int, default=None,
                     help="the seed to use to initialise the model" )
args = parser.parse_args()

# models
# memristor_controller = eval( args.memristor_array )

# hyperparameters
neurons = args.neurons
simulation_time = args.simulation_time
learning_time = args.learning_time
simulation_step = args.simulation_step
function_to_learn = eval( args.function )
input_period = args.input_period
input_frequency = 1 / input_period
pre_nrn = neurons
post_nrn = neurons
err_nrn = neurons
seed = args.seed

with nengo.Network() as model:
    inp = nengo.Node( lambda t: np.sin( input_frequency * 2 * np.pi * t ) )
    pre = nengo.Ensemble(
            n_neurons=pre_nrn,
            dimensions=1,
            encoders=generate_encoders( pre_nrn ),
            intercepts=[ 0.1 ] * pre_nrn,
            label="Pre"
            )
    post = nengo.Ensemble(
            n_neurons=post_nrn,
            dimensions=1,
            encoders=generate_encoders( post_nrn ),
            intercepts=[ 0.1 ] * post_nrn,
            label="Post"
            )
    memr_arr = MemristorArray(
            model=partial( MemristorPair, model=MemristorAnouk ),
            learning_rule=mBCM(),
            in_size=pre_nrn,
            out_size=post_nrn
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
    nengo.Connection( inp, gate_inp_to_post, function=function_to_learn, synapse=None )
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
    
    # plot_network( model )

with nengo.Simulator( model, dt=simulation_step, optimize=True ) as sim:
    sim.run( simulation_time )

plot_ensemble_spikes( sim, "Pre", pre_spikes_probe, pre_probe )
plot_ensemble_spikes( sim, "Post", post_spikes_probe, post_probe )
plot_pre_post( sim, pre_probe, post_probe, inp_probe, time=learning_time )
if neurons <= 10:
    memr_arr.plot_state( sim, "conductance", combined=True )
    for t in range( 0, int( learning_time + 1 ), 1 ):
        memr_arr.plot_weight_matrix( time=t )

print( "Mean squared error:", mse( sim, inp_probe, post_probe, learning_time, simulation_step ) )
print( f"Starting sparsity: {sparsity_measure( memr_arr.get_history( 'weight' )[ 0 ] )}" )
print( f"Ending sparsity: {sparsity_measure( memr_arr.get_history( 'weight' )[ -1 ] )}" )
