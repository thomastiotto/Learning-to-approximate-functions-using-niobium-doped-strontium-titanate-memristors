import matplotlib.pyplot as plt
import numpy as np
import nengo

from nengo.learning_rules import PES
from memristor_nengo.learning_rules import mPES

n_neurons = 4
dimensions = 1
sim_time = 30
learn_time = int( sim_time * 1 / 2 )
seed = 0

model = nengo.Network( seed=seed )
with model:
    # Create an input node
    input_node = nengo.Node(
            # output=lambda t: (np.sin( 1 / 4 * 2 * np.pi * t ), np.sin( 1 / 4 * 2 * np.pi * t + np.pi )),
            # output=nengo.processes.WhiteSignal( 60, high=0.1 ),
            # output=lambda t: int( 6 * t / 5 ) / 3.0 % 2 - 1,
            output=lambda t: np.sin( 1 / 4 * 2 * np.pi * t )
            )
    
    # Shut off learning by inhibiting the error population
    stop_learning = nengo.Node( output=lambda t: t >= learn_time )
    
    # Create the ensemble to represent the input, the learned output, and the error
    pre = nengo.Ensemble( n_neurons, dimensions=dimensions, seed=seed )
    post = nengo.Ensemble( n_neurons, dimensions=dimensions, seed=seed )
    error = nengo.Ensemble( n_neurons, dimensions=dimensions, radius=2, seed=seed )
    
    # Connect pre and post with a communication channel
    conn = nengo.Connection(
            pre.neurons,
            post.neurons,
            # transform=mPES.initial_normalized_conductances( 1e8, 1e8, (post.n_neurons, pre.n_neurons), seed=seed ),
            transform=np.zeros( (post.n_neurons, pre.n_neurons) )
            )
    # the matrix given to transform are the initial weights found in model.sig[conn]["weights"]
    
    # Apply the mPES learning rule to conn
    conn.learning_rule_type = mPES()
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
    if isinstance( conn.learning_rule_type, mPES ):
        pos_memr_probe = nengo.Probe( conn.learning_rule, "pos_memristors", synapse=None )
        neg_memr_probe = nengo.Probe( conn.learning_rule, "neg_memristors", synapse=None )

# Create the simulator
with nengo.Simulator( model ) as sim:
    sim.run( sim_time )

# Plot the input signal
plt.figure( figsize=(12, 8) )
plt.subplot( 3, 1, 1 )
plt.plot(
        sim.trange(),
        sim.data[ input_node_probe ],
        label='Input',
        linewidth=2.0 )
plt.plot(
        sim.trange(),
        sim.data[ learn_probe ],
        label='Stop learning?',
        color='r',
        linewidth=2.0 )
plt.legend( loc='lower left' )
plt.ylim( -1.2, 1.2 )
plt.subplot( 3, 1, 2 )
plt.plot(
        sim.trange(),
        sim.data[ pre_probe ],
        label='Input',
        linewidth=2.0 )
plt.plot(
        sim.trange(),
        sim.data[ post_probe ],
        label='Post' )
plt.legend( loc='lower left' )
plt.ylim( -1.2, 1.2 )
plt.subplot( 3, 1, 3 )
plt.plot(
        sim.trange(),
        sim.data[ error_probe ],
        label='Error' )
plt.legend( loc='lower left' )
plt.tight_layout()
plt.show()

plt.figure( figsize=(12, 8) )
plt.subplot( 2, 1, 1 )
plt.plot( sim.trange(), sim.data[ pre_probe ], label="Pre" )
plt.plot( sim.trange(), sim.data[ post_probe ], label="Post" )
plt.ylabel( "Decoded value" )
plt.ylim( -1.6, 1.6 )
plt.legend( loc="lower left" )
plt.subplot( 2, 1, 2 )
# Find weight row with max variance
neuron = np.argmax( np.mean( np.var( sim.data[ weight_probe ], axis=0 ), axis=1 ) )
plt.plot( sim.trange(), sim.data[ weight_probe ][ ..., neuron ] )
plt.ylabel( "Connection weight" )
plt.show()

n_cols = 5
n_rows = int( learn_time / 5 )
plt.figure( figsize=(12, 8) )
fig, axes = plt.subplots( n_rows, n_cols )
t = 0
for i in range( axes.shape[ 0 ] ):
    for j in range( axes.shape[ 1 ] ):
        axes[ i, j ].matshow( sim.data[ weight_probe ][ int( t / sim.dt ), ... ], cmap=plt.cm.Blues )
        axes[ i, j ].set_title( f"{t}" )
        print( t )
        print( sim.data[ weight_probe ][ int( t / sim.dt ), ... ] )
        t += 1
plt.suptitle( "Weights over learning" )
plt.tight_layout()
plt.show()

plt.figure( figsize=(12, 8) )
for i in range( n_neurons ):
    plt.plot( sim.trange(), sim.data[ pos_memr_probe ][ ..., i ], c="r" )
    plt.plot( sim.trange(), sim.data[ neg_memr_probe ][ ..., i ], c="b" )
    plt.ylabel( "Memristor resistances" )
plt.show()

print( np.average( sim.data[ weight_probe ][ int( learn_time / sim.dt ), ... ] ) )
