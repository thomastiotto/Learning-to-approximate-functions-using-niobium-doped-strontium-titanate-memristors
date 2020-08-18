import nengo
import nengo_dl
from sklearn.metrics import mean_squared_error
import time
import argparse

from nengo.processes import WhiteSignal
from nengo.learning_rules import PES
from memristor_nengo.learning_rules import mPES
from memristor_nengo.extras import *

parser = argparse.ArgumentParser()
parser.add_argument( "-f", "--function", default="lambda x: x" )
parser.add_argument( "-O", "--output", default="generate_sines( dimensions )" )
parser.add_argument( "-t", "--timestep", default=0.001, type=int )
parser.add_argument( "-S", "--simulation_time", default=30, type=int )
parser.add_argument( "-N", "--neurons", nargs="*", default=[ 10, 10, 10 ], type=int )
parser.add_argument( "-d", "--dimensions", default=3, type=int )
parser.add_argument( "-n", "--noise", default=0.15, type=float )
parser.add_argument( "-l", "--learning_rule", default="mPES", choices=[ "mPES", "PES" ] )
parser.add_argument( "-b", "--backend", default="nengo_dl", choices=[ "nengo_dl", "nengo_core" ] )
parser.add_argument( "-o", "--optimisations", default="run", choices=[ "run", "build", "memory" ] )
parser.add_argument( "-s", "--seed", default=None, type=int )
parser.add_argument( "-p", "--plot", action="count", default=0 )
parser.add_argument( "-v", "--verbosity", action="count", default=0 )
parser.add_argument( "-pd", "--plots_directory", default="../data/" )

args = parser.parse_args()
if args.neurons is not None and len( args.neurons ) not in (0, 3):
    parser.error( 'Either give no values for action, or three, not {}.'.format( len( args.neurons ) ) )

# TODO read parameters from conf file https://docs.python.org/3/library/configparser.html
function_to_learn = eval( args.function )
timestep = args.timestep
sim_time = args.simulation_time
pre_n_neurons = args.neurons[ 0 ]
post_n_neurons = args.neurons[ 1 ]
error_n_neurons = args.neurons[ 2 ]
dimensions = args.dimensions
noise_percent = args.noise
learning_rule = args.learning_rule
backend = args.backend
optimisations = args.optimisations
seed = args.seed
generate_plots = show_plots = save_plots = False
if args.plot >= 1:
    generate_plots = True
    show_plots = True
    if args.plot == 2:
        save_plots = True
verboseprint = print if args.verbosity >= 1 else lambda *a, **k: None
plots_directory = args.plots_directory

learn_time = int( sim_time * 3 / 4 )
n_neurons = np.amax( [ pre_n_neurons, post_n_neurons ] )
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
verboseprint( f"Using {optimisations} optimisation" )

model = nengo.Network( seed=seed )
with model:
    # Create an input node
    input_node = nengo.Node(
            output=eval( args.output ),
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
    
    # Connect pre and post with a communication channel
    # the matrix given to transform are the initial weights found in model.sig[conn]["weights"]
    conn = nengo.Connection(
            pre.neurons,
            post.neurons,
            transform=np.zeros( (post.n_neurons, pre.n_neurons) )
            )
    
    # Apply the learning rule to conn
    if learning_rule == "mPES":
        conn.learning_rule_type = mPES(
                noisy=noise_percent,
                seed=seed )
    if learning_rule == "PES":
        conn.learning_rule_type = PES()
    verboseprint( "Simulating with", conn.learning_rule_type )
    
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
verboseprint( f"Backend is {backend}" )
if backend == "nengo_core":
    cm = nengo.Simulator( model, seed=seed, dt=timestep, optimize=optimize )
if backend == "nengo_dl":
    cm = nengo_dl.Simulator( model, seed=seed, dt=timestep )
start_time = time.time()
with cm as sim:
    for i in range( simulation_discretisation ):
        verboseprint( f"\nRunning discretised step {i + 1} of {simulation_discretisation}" )
        sim.run( sim_time / simulation_discretisation )
verboseprint( f"\nTotal time for simulation: {time.strftime( '%H:%M:%S', time.gmtime( time.time() - start_time ) )} s" )

verboseprint( "Weights average after learning:" )
verboseprint( np.average( sim.data[ weight_probe ][ -1, ... ] ) )
verboseprint( "Weights sparsity at t=0 and after learning:" )
verboseprint( gini( sim.data[ weight_probe ][ 0 ] ), end=" -> " )
verboseprint( gini( sim.data[ weight_probe ][ -1 ] ) )
verboseprint( "MSE after learning [f(pre) vs. post]:" )
mse = mean_squared_error(
        function_to_learn( sim.data[ pre_probe ][ int( (learn_time / sim.dt) / (sample_every / timestep) ):, ... ] ),
        sim.data[ post_probe ][ int( (learn_time / sim.dt) / (sample_every / timestep) ):, ... ]
        )
verboseprint( mse )

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
        
        dir_name, dir_images, dir_data = make_timestamped_dir( root=plots_directory + learning_rule + "/" )
        
        save_weights( dir_data, sim.data[ weight_probe ] )
        print( f"Saved NumPy weights in {dir_data}" )
        
        save_results_to_csv( dir_data, sim.data[ input_node_probe ], sim.data[ pre_probe ], sim.data[ post_probe ],
                             sim.data[ post_probe ] - function_to_learn( sim.data[ pre_probe ] ) )
        save_memristors_to_csv( dir_data, sim.data[ pos_memr_probe ], sim.data[ neg_memr_probe ] )
        print( f"Saved data in {dir_data}" )
        
        for i, fig in enumerate( plots ):
            fig.savefig( dir_images + str( i ) + ".pdf" )
            fig.savefig( dir_images + str( i ) + ".png" )
        
        print( f"Saved plots in {dir_images}" )
    
    
    save_plots()

if show_plots:
    def show_plots():
        assert generate_plots
        
        for i, fig in enumerate( plots ):
            fig.show()
    
    
    show_plots()
