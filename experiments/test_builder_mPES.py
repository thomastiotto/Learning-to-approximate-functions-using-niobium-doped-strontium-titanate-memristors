import nengo
import nengo_dl
from sklearn.metrics import mean_squared_error
import time

from nengo.processes import WhiteSignal
from nengo.learning_rules import PES
from memristor_nengo.learning_rules import mPES
from memristor_nengo.extras import *

function_to_learn = lambda x: x
timestep = 0.001
sim_time = 30
pre_n_neurons = 100
post_n_neurons = 100
error_n_neurons = 100
dimensions = 3
noise_percent = 0.15
learning_rule = "mPES"
backend = "nengo_dl"
optimisations = "run"
seed = None
generate_plots = True
show_plots = True
save_plots = False

learn_time = int( sim_time * 3 / 4 )
n_neurons = np.amax( [ pre_n_neurons, post_n_neurons ] )

model = nengo.Network( seed=seed )
with model:
    if optimisations == "build":
        optimize = False
        sample_every = timestep
        simulation_discretisation = 1
    elif optimisations == "run":
        optimize = True
        sample_every = timestep
        simulation_discretisation = 1
    elif optimisations == "memory":
        optimize = False
        sample_every = timestep * 100
        simulation_discretisation = n_neurons
    print( f"Using {optimisations} optimisation" )
    
    # Create an input node
    input_node = nengo.Node(
            output=generate_sines( dimensions ),
            # output=WhiteSignal( 60, high=5, seed=seed ),
            size_out=dimensions
            )
    
    # Shut off learning by inhibiting the error population
    stop_learning = nengo.Node( output=lambda t: t >= learn_time )
    
    # Create the ensemble to represent the input, the learned output, and the error
    pre = nengo.Ensemble( pre_n_neurons, dimensions=dimensions, seed=seed,
                          # encoders=[ [ 1. ], [ -1. ], [ -1. ], [ -1. ] ]
                          )
    post = nengo.Ensemble( post_n_neurons, dimensions=dimensions, seed=seed,
                           # encoders=[ [ 1. ], [ -1. ], [ -1. ], [ -1. ] ]
                           )
    error = nengo.Ensemble( error_n_neurons, dimensions=dimensions, radius=2, seed=seed )
    
    # Connect pre and post with exponent communication channel
    conn = nengo.Connection(
            pre.neurons,
            post.neurons,
            transform=np.zeros( (post.n_neurons, pre.n_neurons) )
            )
    # the matrix given to transform are the initial weights found in model.sig[conn]["weights"]
    
    # Apply the mPES learning rule to conn
    if learning_rule == "mPES":
        conn.learning_rule_type = mPES(
                noisy=noise_percent,
                seed=seed )
    if learning_rule == "PES":
        conn.learning_rule_type = PES()
    print( "Simulating with", conn.learning_rule_type )
    
    # Provide an error signal to the learning rule
    nengo.Connection( error, conn.learning_rule )
    
    # Compute the error signal (error = actual - target)
    nengo.Connection( post, error )
    
    # Subtract the target (this would normally come from some external system)
    nengo.Connection( pre, error, function=function_to_learn, transform=-1 )
    
    # Connect the input node to ensemble pre
    nengo.Connection( input_node, pre )
    
    nengo.Connection(
            stop_learning,
            error.neurons,
            transform=-20 * np.ones( (error.n_neurons, 1) ) )
    
    input_node_probe = nengo.Probe( input_node, sample_every=sample_every )
    pre_probe = nengo.Probe( pre, synapse=0.01, sample_every=sample_every )
    post_probe = nengo.Probe( post, synapse=0.01, sample_every=sample_every )
    error_probe = nengo.Probe( error, synapse=0.01, sample_every=sample_every )
    learn_probe = nengo.Probe( stop_learning, synapse=None, sample_every=sample_every )
    weight_probe = nengo.Probe( conn, "weights", synapse=None, sample_every=sample_every )
    post_spikes_probe = nengo.Probe( post.neurons, sample_every=sample_every )
    if isinstance( conn.learning_rule_type, mPES ):
        pos_memr_probe = nengo.Probe( conn.learning_rule, "pos_memristors", synapse=None, sample_every=sample_every )
        neg_memr_probe = nengo.Probe( conn.learning_rule, "neg_memristors", synapse=None, sample_every=sample_every )

# Create the simulator
print( f"Backend is {backend}" )
if backend == "nengo":
    cm = nengo.Simulator( model, seed=seed, dt=timestep, optimize=optimize )
if backend == "nengo_dl":
    cm = nengo_dl.Simulator( model, seed=seed, dt=timestep )
start_time = time.time()
with cm as sim:
    for i in range( simulation_discretisation ):
        print( f"\nRunning discretised step {i + 1} of {simulation_discretisation}" )
        sim.run( sim_time / simulation_discretisation )
print( f"\nTotal time for simulation: {time.strftime( '%H:%M:%S', time.gmtime( time.time() - start_time ) )} s" )

print( "Weights average after learning:" )
print( np.average( sim.data[ weight_probe ][ -1, ... ] ) )
print( "Weights sparsity at t=0 and after learning:" )
print( gini( sim.data[ weight_probe ][ 0 ] ), end=" -> " )
print( gini( sim.data[ weight_probe ][ -1 ] ) )
print( "MSE after learning [f(pre) vs. post]:" )
mse = mean_squared_error(
        function_to_learn( sim.data[ pre_probe ][ int( (learn_time / sim.dt) / (sample_every / timestep) ):, ... ] ),
        sim.data[ post_probe ][ int( (learn_time / sim.dt) / (sample_every / timestep) ):, ... ]
        )
print( mse )

plots = [ ]
if generate_plots:
    def generate_plots():
        plotter = Plotter( sim.trange( sample_every=sample_every ), post_n_neurons, pre_n_neurons, dimensions,
                           learn_time,
                           plot_size=(13, 7),
                           dpi=300 )
        
        plots.append(
                plotter.plot_results( sim.data[ input_node_probe ], sim.data[ pre_probe ], sim.data[ post_probe ],
                                      error=sim.data[ post_probe ] - function_to_learn( sim.data[ pre_probe ] ),
                                      smooth=True,
                                      mse=mse )
                )
        plots.append(
                plotter.plot_ensemble_spikes( "Post", sim.data[ post_spikes_probe ], sim.data[ post_probe ] )
                )
        plots.append(
                plotter.plot_weight_matrices_over_time( sim.data[ weight_probe ], sample_every=sample_every )
                )
        if n_neurons <= 10:
            plots.append(
                    plotter.plot_weights_over_time( sim.data[ pos_memr_probe ], sim.data[ neg_memr_probe ] )
                    )
            plots.append(
                    plotter.plot_values_over_time( 1 / sim.data[ pos_memr_probe ], 1 / sim.data[ neg_memr_probe ] )
                    )
    
    
    generate_plots()

if save_plots:
    def save_plots():
        assert generate_plots
        
        dir_name, dir_images = make_timestamped_dir( root="../data/mPES/" )
        save_weights( dir_name, sim.data[ weight_probe ] )
        for i, fig in enumerate( plots ):
            fig.savefig( dir_images + str( i ) + ".pdf" )
            fig.savefig( dir_images + str( i ) + ".png" )
    
    
    save_plots()

if show_plots:
    def show_plots():
        assert generate_plots
        
        for i, fig in enumerate( plots ):
            fig.show()
    
    
    show_plots()
