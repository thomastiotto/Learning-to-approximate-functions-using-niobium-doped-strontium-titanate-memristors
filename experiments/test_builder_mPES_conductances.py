import matplotlib.pyplot as plt
import nengo
from sklearn.metrics import mean_squared_error

from nengo.learning_rules import PES
from memristor_nengo.learning_rules_conductances import mPES
from memristor_nengo.extras import *

n_neurons = 4
dimensions = 1
sim_time = 30
learn_time = int( sim_time * 3 / 4 )
seed = None

model = nengo.Network( seed=seed )
with model:
    # Create an input node
    input_node = nengo.Node(
            output=generate_sines( dimensions )
            )
    
    # Shut off learning by inhibiting the error population
    stop_learning = nengo.Node( output=lambda t: t >= learn_time )
    
    # Create the ensemble to represent the input, the learned output, and the error
    pre = nengo.Ensemble( n_neurons, dimensions=dimensions, seed=seed,
                          # encoders=[ [ 1. ], [ -1. ], [ -1. ], [ -1. ] ]
                          )
    post = nengo.Ensemble( n_neurons, dimensions=dimensions, seed=seed,
                           # encoders=[ [ 1. ], [ -1. ], [ -1. ], [ -1. ] ]
                           )
    error = nengo.Ensemble( n_neurons, dimensions=dimensions, radius=2, seed=seed )
    
    
    def initial_resistances( low, high, shape ):
        return np.random.uniform( low, high, shape )
    
    
    initial_conductances_pos = 1 / initial_resistances( 1e8, 1.1e8, (post.n_neurons, pre.n_neurons) )
    initial_conductances_neg = 1 / initial_resistances( 1e8, 1.1e8, (post.n_neurons, pre.n_neurons) )
    initial_weights = (initial_conductances_pos - initial_conductances_neg) * 1e5
    
    # Connect pre and post with exponent communication channel
    conn = nengo.Connection(
            pre.neurons,
            post.neurons,
            # transform=np.zeros( (post.n_neurons, pre.n_neurons) )
            transform=initial_weights
            )
    # the matrix given to transform are the initial weights found in model.sig[conn]["weights"]
    
    # Apply the mPES learning rule to conn
    conn.learning_rule_type = mPES( initial_conductances_pos, initial_conductances_neg, seed=seed )
    # conn.learning_rule_type = PES()
    print( "Simulating with", conn.learning_rule_type )
    
    # Provide an error signal to the learning rule
    nengo.Connection( error, conn.learning_rule )
    
    # Compute the error signal (error = actual - target)
    nengo.Connection( post, error )
    
    # Subtract the target (this would normally come from some external system)
    nengo.Connection( pre, error, function=lambda x: x, transform=-1 )
    
    # Connect the input node to ensemble pre
    nengo.Connection( input_node, pre )
    
    nengo.Connection(
            stop_learning,
            error.neurons,
            transform=-20 * np.ones( (error.n_neurons, 1) ) )
    
    input_node_probe = nengo.Probe( input_node )
    pre_probe = nengo.Probe( pre, synapse=0.01 )
    post_probe = nengo.Probe( post, synapse=0.01 )
    error_probe = nengo.Probe( error, synapse=0.01 )
    learn_probe = nengo.Probe( stop_learning, synapse=None )
    weight_probe = nengo.Probe( conn, "weights", synapse=None )
    post_spikes_probe = nengo.Probe( post.neurons )
    if isinstance( conn.learning_rule_type, mPES ):
        pos_memr_probe = nengo.Probe( conn.learning_rule, "pos_memristors", synapse=None )
        neg_memr_probe = nengo.Probe( conn.learning_rule, "neg_memristors", synapse=None )

# Create the simulator
with nengo.Simulator( model ) as sim:
    sim.run( sim_time )

set_plot_params( sim.trange(), post.n_neurons, pre.n_neurons )
plot_results( sim.data[ input_node_probe ], sim.data[ learn_probe ], sim.data[ pre_probe ], sim.data[ post_probe ],
              sim.data[ error_probe ], smooth=True )
plt.show()
plot_ensemble_spikes( "Post", sim.data[ post_spikes_probe ], sim.data[ post_probe ] )
plt.show()
plot_weight_matrices_over_time( 5, learn_time, sim.data[ weight_probe ], sim.dt )
plt.show()
plot_weights_over_time( sim.data[ pos_memr_probe ], sim.data[ neg_memr_probe ] )
plt.show()
plot_values_over_time( sim.data[ pos_memr_probe ], sim.data[ neg_memr_probe ] )
plt.show()

print( "Final weights average:" )
print( np.average( sim.data[ weight_probe ][ -1, ... ] ) )
print( "MSE (input vs. post):" )
print( mean_squared_error( sim.data[ pre_probe ][ int( learn_time / sim.dt ):, ... ],
                           sim.data[ post_probe ][ int( learn_time / sim.dt ):, ... ]
                           )
       )
