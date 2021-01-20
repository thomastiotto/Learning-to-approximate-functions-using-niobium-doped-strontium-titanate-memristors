import matplotlib.pyplot as plt
from random import randrange
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random
import argparse
import multiprocessing as mp
import shutil
import tempfile

import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

from nengo.utils.matplotlib import rasterplot
from memristor_nengo.extras import *
from memristor_nengo.neurons import *

parser = argparse.ArgumentParser()
parser.add_argument( "-S", "--train_samples", default=None, type=int,
                     help="The number of samples to train/test on.  Default is all dataset" )
parser.add_argument( "-D", "--digits", nargs="*", default=None, action="store", type=int,
                     help="The digits to train on.  Default is all digits" )
parser.add_argument( "-N", "--neurons", default=10, type=int,
                     help="The number of excitatory neurons.  Default is 10" )
parser.add_argument( "--learning_rate", default=1e-6, type=float,
                     help="Learning rate in Oja.  Default is 1e-6" )
parser.add_argument( "--beta", default=1, type=float,
                     help="Beta in Oja. How strong forgetting is.  Default is 1" )
parser.add_argument( "--tau_inh", default=10, type=int,
                     help="Lateral inhibition. The number of timesteps the losing neurons will be inhibited for.  "
                          "Default is 10" )
parser.add_argument( "--inc_n", default=0.01, type=float,
                     help="Adaptation increment. How much the adaptation state is increased after each spike.  "
                          "Default is 0.01" )
parser.add_argument( "--tau_n", default=1, type=float,
                     help="Adaptation time constant. Affects how quickly the adaptation state decays to zero in the "
                          "absence of spikes \(larger = slower decay\).  Default is 1" )
parser.add_argument( "-s", "--seed", default=None, type=int )
parser.add_argument( "-b", "--backend", default="nengo_core",
                     choices=[ "nengo_dl", "nengo_core" ] )
parser.add_argument( "-d", "--device", default="/cpu:0",
                     help="/cpu:0 or /gpu:[x]" )
parser.add_argument( "-pd", "--plots_directory", default="../data/MNIST/",
                     help="Directory where plots will be saved.  Default is ../data/" )
parser.add_argument( '--video', dest='video', action='store_true' )
parser.add_argument( '--no-video', dest='video', action='store_false' )
parser.add_argument( '--only-video', dest='only_video', action='store_true' )
parser.set_defaults( feature=True )
args = parser.parse_args()

# load mnist dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# change inputs to [0-1] range
train_images = train_images / 255
test_images = test_images / 255

# reshape the labels to rank 3 (as expected in Nengo)
train_labels = train_labels[ :, None, None ]
test_labels = test_labels[ :, None, None ]

train_test_proportion = train_images.shape[ 0 ] / test_images.shape[ 0 ]

# set parameters
digits = args.digits
num_samples = args.train_samples
post_n_neurons = args.neurons
random.seed = args.seed
presentation_time = 0.35
pause_time = 0.15

if digits:
    train_images = np.array(
            [ x for i, x in enumerate( train_images ) if train_labels[ i ] in digits ]
            )
    test_images = np.array(
            [ x for i, x in enumerate( test_images ) if test_labels[ i ] in digits ]
            )
    train_labels = np.array(
            [ x for i, x in enumerate( train_labels ) if train_labels[ i ] in digits ]
            )
    test_labels = np.array(
            [ x for i, x in enumerate( test_labels ) if test_labels[ i ] in digits ]
            )
if num_samples:
    train_images = train_images[ :num_samples ]
    test_images = test_images[ :int( num_samples / train_test_proportion ) ]
    train_labels = train_labels[ :num_samples ]
    test_labels = test_labels[ :int( num_samples / train_test_proportion ) ]

num_train_samples = train_images.shape[ 0 ]
num_test_samples = test_images.shape[ 0 ]
dt = 0.001
sample_every = 100 * dt
args.video = True if args.only_video or args.video else False
sim_train_time = (presentation_time) * train_images.shape[ 0 ]
sim_test_time = (presentation_time) * test_images.shape[ 0 ]
# sim_train_time = (presentation_time + pause_time) * train_images.shape[ 0 ]
# sim_test_time = (presentation_time + pause_time) * test_images.shape[ 0 ]
sample_every_weights = num_samples * dt if args.video else sim_train_time

print( "######################################################",
       "###################### DEFINITION ####################",
       "######################################################",
       sep="\n" )

model = nengo.Network( seed=args.seed )
with model:
    setup()
    
    inp = nengo.Node( nengo.processes.PresentInput( train_images, presentation_time ) )
    # inp = nengo.Node( PresentInputWithPause( train_images, presentation_time, pause_time ) )
    pre = nengo.Ensemble( n_neurons=784, dimensions=1,
                          neuron_type=nengo.neurons.PoissonSpiking( nengo.LIFRate() ),
                          encoders=nengo.dists.Choice( [ [ 1 ] ] ),
                          intercepts=nengo.dists.Choice( [ 0 ] ),
                          # max_rates=nengo.dists.Choice( [ 20, 22 ] )
                          seed=args.seed
                          )
    post = nengo.Ensemble( n_neurons=post_n_neurons, dimensions=1,
                           neuron_type=AdaptiveLIFLateralInhibition( inc_n=args.inc_n,
                                                                     tau_n=args.tau_n,
                                                                     tau_inhibition=args.tau_inh,
                                                                     reset_every=presentation_time ),
                           # neuron_type=nengo.neurons.AdaptiveLIF(),
                           encoders=nengo.dists.Choice( [ [ 1 ] ] ),
                           intercepts=nengo.dists.Choice( [ 0 ] ),
                           max_rates=nengo.dists.Choice( [ 20, 22 ] ),
                           seed=args.seed
                           )
    
    nengo.Connection( inp, pre.neurons )
    
    conn = nengo.Connection( pre.neurons, post.neurons,
                             learning_rule_type=nengo.learning_rules.Oja( learning_rate=args.learning_rate,
                                                                          beta=args.beta ),
                             transform=np.random.random( (post.n_neurons, pre.n_neurons) )
                             )
    
    pre_probe = nengo.Probe( pre.neurons, sample_every=sample_every )
    post_probe = nengo.Probe( post.neurons, sample_every=dt )
    weight_probe = nengo.Probe( conn, "weights", sample_every=sample_every_weights )
    adaptation_probe = nengo.Probe( post.neurons, "adaptation", sample_every=sim_train_time )

un_train, cnt_train = np.unique( train_labels, return_counts=True )
un_test, cnt_test = np.unique( test_labels, return_counts=True )

from ascii_graph import Pyasciigraph

graph = Pyasciigraph( force_max_value=100 )

print( f"Samples: {num_samples}" )
for line in graph.graph( "Train digits distribution",
                         [ (str( x ), c / len( train_labels ) * 100) for x, c in zip( un_train, cnt_train ) ] ):
    print( line )
for line in graph.graph( "Test digits distribution",
                         [ (str( x ), c / len( test_labels ) * 100) for x, c in zip( un_test, cnt_test ) ] ):
    print( line )
print( "Pre:\n\t", pre.neuron_type, "\n\tNeurons:", pre.n_neurons, "\n\tRate:", pre.max_rates )
print( "Post:\n\t", post.neuron_type, "\n\tNeurons:", post.n_neurons, "\n\tRate:", post.max_rates )
print( "Rule:\n\t", conn.learning_rule_type )

print( "######################################################",
       "####################### TRAINING #####################",
       "######################################################",
       sep="\n" )

print( f"Backend is {args.backend}, running on ", end="" )
if args.backend == "nengo_core":
    print( "CPU" )
    with nengo.Simulator( model, seed=args.seed ) as sim_train:
        sim_train.run( sim_train_time )
if args.backend == "nengo_dl":
    print( args.device )
    with nengo_dl.Simulator( model, seed=args.seed, device=args.device ) as sim_train:
        sim_train.run( sim_train_time )

# print number of recorded spikes
num_spikes_train = np.sum( sim_train.data[ post_probe ] > 0, axis=0 )
for line in graph.graph( f"Spikes distribution (timestep={dt}):",
                         [ (str( i ), x / np.sum( num_spikes_train ) * 100) for i, x in
                           enumerate( num_spikes_train ) ] ):
    print( line )
print( "\tTotal:", np.sum( num_spikes_train ) )
print( f"\tNormalised standard dev.: {np.std( num_spikes_train ) / np.mean( num_spikes_train )}" )

fig1, ax = plt.subplots()
rasterplot( sim_train.trange( sample_every=sample_every ), sim_train.data[ pre_probe ], ax )
ax.set_ylabel( 'Neuron' )
ax.set_xlabel( 'Time (s)' )
fig1.get_axes()[ 0 ].annotate( "Pre" + " neural activity", (0.5, 0.94),
                               xycoords='figure fraction', ha='center',
                               fontsize=20
                               )
fig1.show()

fig2, ax = plt.subplots()
rasterplot( sim_train.trange( sample_every=dt ), sim_train.data[ post_probe ], ax )
ax.set_ylabel( 'Neuron' )
ax.set_xlabel( 'Time (s)' )
fig2.get_axes()[ 0 ].annotate( "Post" + " neural activity", (0.5, 0.94),
                               xycoords='figure fraction', ha='center',
                               fontsize=20
                               )
fig2.show()

# TODO sample neurons if too many to show
# TODO not working with neurons not multiple of 10
plt.set_cmap( 'jet' )
fig3, axes = plt.subplots( int( post.n_neurons / 10 ), 10 )
for i, ax in enumerate( axes.flatten() ):
    ax.matshow( sim_train.data[ weight_probe ][ -1, i, ... ].reshape( (28, 28) ) )
    ax.set_title( f"N. {i}" )
    ax.set_yticks( [ ] )
    ax.set_xticks( [ ] )
fig3.suptitle( "Weights after learning" )
fig3.tight_layout()
fig3.show()

# allocate folder to save outputs
dir_name, dir_images, dir_data = make_timestamped_dir( root=args.plots_directory )

# generate heatmap evolution video
if args.video:
    if __name__ == "__main__":
        mp.set_start_method( "fork" )
        p = mp.Process( target=generate_heatmap, args=(sim_train.data[ weight_probe ], dir_images) )
        p.start()

# write definition to temp file location because we modify the network later on
tf = tempfile.NamedTemporaryFile( mode='w+t', delete=False )
with tf as f:
    f.write( "\n###################### DEFINITION ####################\n" )
    f.write( f"Samples: {num_samples}\n" )
    for line in graph.graph( "Train digits distribution",
                             [ (str( x ), c / len( train_labels ) * 100) for x, c in zip( un_train, cnt_train ) ] ):
        f.write( f"{line}\n" )
    for line in graph.graph( "Test digits distribution",
                             [ (str( x ), c / len( test_labels ) * 100) for x, c in zip( un_test, cnt_test ) ] ):
        f.write( f"{line}\n" )
    f.write( f"Pre:\n\t {pre.neuron_type} \n\tNeurons: {pre.n_neurons} \n\tRate: {pre.max_rates}\n" )
    f.write( f"Post:\n\t {post.neuron_type} \n\tNeurons: {post.n_neurons} \n\tRate: {post.max_rates}\n" )
    f.write( f"Rule:\n\t {conn.learning_rule_type}\n" )

if not args.only_video:
    print( "######################################################",
           "################### CLASS ASSIGNMENT #################",
           "######################################################",
           sep="\n" )
    # load weights found during training and freeze them
    conn.transform = sim_train.data[ weight_probe ][ -1 ].squeeze()
    conn.learning_rule_type = nengo.learning_rules.Oja( learning_rate=0 )
    # load last neuron thresholds from training and freeze them
    post.neuron_type = AdaptiveLIFLateralInhibition(
            tau_n=1e20,
            inc_n=0,
            initial_state={
                    "adaptation": sim_train.data[ adaptation_probe ].squeeze() }
            )
    
    # set post probe to record every spike for statistics
    post_probe.sample_every = dt
    
    if args.backend == "nengo_core":
        with nengo.Simulator( model, seed=args.seed ) as sim_class:
            sim_class.run( sim_train_time )
    if args.backend == "nengo_dl":
        with nengo_dl.Simulator( model, seed=args.seed, device=args.device ) as sim_class:
            sim_class.run( sim_train_time )
    
    # print number of recorded spikes
    num_spikes_class = np.sum( sim_class.data[ post_probe ] > 0, axis=0 )
    for line in graph.graph( f"Spikes distribution (timestep={dt}):",
                             [ (str( i ), x / np.sum( num_spikes_train ) * 100) for i, x in
                               enumerate( num_spikes_class ) ] ):
        print( line )
    print( "\tTotal:", np.sum( num_spikes_class ) )
    print( f"\tNormalised standard dev.: {np.std( num_spikes_class ) / np.mean( num_spikes_class )}" )
    
    # remove spikes that happened during pause period
    mask = np.concatenate( (np.ones( int( presentation_time / dt ) + 1 ), np.zeros( int( pause_time / dt ) )) ) \
        .astype( bool )
    clean_post_spikes_class = sim_class.data[ post_probe ].reshape( num_train_samples, -1, post_n_neurons )
    # clean_post_spikes_class = sim_class.data[ post_probe ].reshape( num_train_samples, -1, post_n_neurons )[ :, mask,
    #                           ... ]
    
    # count neuron activations in response to each example
    neuron_activations_class = np.count_nonzero( clean_post_spikes_class, axis=1 )
    
    # count how many times each neuron spiked for each label across samples
    neuron_label_count = { neur: { lab: 0 for lab in un_train } for neur in range( post_n_neurons ) }
    for t, lab in enumerate( train_labels.ravel() ):
        for neur in range( post_n_neurons ):
            neuron_label_count[ neur ][ lab ] += neuron_activations_class[ t, neur ]
    print( "Neuron activations for each label:\n", end="" )
    pprint_dict( neuron_label_count, level=1 )
    
    # associate each neuron with the label it spiked most for
    # if the neuron never spiked pick a random label
    neuron_label = { neur: max( lab, key=lab.get ) if any( lab.values() ) else randrange(
            len( un_train ) ) for neur, lab in neuron_label_count.items() }
    print( "Label associated to each neuron:\n", end="" )
    pprint_dict( neuron_label, level=1 )
    
    # group neurons into classes based on their label
    label_class = { }
    for neur, lab in neuron_label.items():
        label_class.setdefault( lab, list() ).append( neur )
    print( "Neuron set associated to each label:\n", end="" )
    pprint_dict( label_class, level=1 )
    print( f"Number of labels discovered: {len( label_class.keys() )}" )
    
    print( "######################################################",
           "###################### INFERENCE #####################",
           "######################################################",
           sep="\n" )
    
    # switch to test set
    inp.output = nengo.processes.PresentInput( test_images, presentation_time )
    # inp.output = PresentInputWithPause( test_images, presentation_time, pause_time )
    
    if args.backend == "nengo_core":
        with nengo.Simulator( model, seed=args.seed ) as sim_test:
            sim_test.run( sim_test_time )
    if args.backend == "nengo_dl":
        with nengo_dl.Simulator( model, seed=args.seed, device=args.device ) as sim_test:
            sim_test.run( sim_test_time )
    
    # print number of recorded spikes
    num_spikes_test = np.sum( sim_test.data[ post_probe ] > 0, axis=0 )
    for line in graph.graph( f"Spikes distribution (timestep={dt}):",
                             [ (str( i ), x / np.sum( num_spikes_train ) * 100) for i, x in
                               enumerate( num_spikes_train ) ] ):
        print( line )
    print( "\tTotal:", np.sum( num_spikes_test ) )
    print( f"\tNormalised standard dev.: {np.std( num_spikes_test ) / np.mean( num_spikes_test )}" )
    
    # remove spikes that happened during pause period
    clean_post_spikes_test = sim_test.data[ post_probe ].reshape( num_test_samples, -1, post_n_neurons )
    # clean_post_spikes_test = sim_test.data[ post_probe ].reshape( num_test_samples, -1, post_n_neurons )[ :, mask,
    # ... ]
    
    # count neuron activations in response to each example
    neuron_activations_test = np.count_nonzero( clean_post_spikes_test, axis=1 )
    
    # use the neuron class with highest average activation as class prediction at each timestep
    prediction = [ ]
    for t in range( num_test_samples ):
        max_mean = 0
        max_lab = None
        for lab, neur in label_class.items():
            mean = np.mean( neuron_activations_test[ t, neur ] )
            if mean > max_mean:
                max_mean = mean
                max_lab = lab
        prediction.append( max_lab )
    
    print( "Classification results:" )
    print( "\tAccuracy:", accuracy_score( test_labels.ravel(), prediction ) )
    print( "\tConfusion matrix:\n", confusion_matrix( test_labels.ravel(), prediction ) )
    
    fig1.savefig( dir_images + "pre" + ".eps" )
    fig2.savefig( dir_images + "post" + ".eps" )
    fig3.savefig( dir_images + "weights" + ".eps" )
    print( f"Saved plots in {dir_images}" )
    
    # move file from temp location and write rest of outputs
    shutil.move( tf.name, dir_data + "results.txt" )
    with open( dir_data + "results.txt", "a" ) as f:
        f.write( "\n####################### TRAINING #####################\n" )
        for line in graph.graph( f"Spikes distribution (timestep={dt}):",
                                 [ (str( i ), x / np.sum( num_spikes_train ) * 100) for i, x in
                                   enumerate( num_spikes_train ) ] ):
            f.write( f"{line}\n" )
        f.write( f"\tTotal: {np.sum( num_spikes_train )}\n" )
        f.write( f"\tNormalised standard dev.: {np.std( num_spikes_train ) / np.mean( num_spikes_train )}\n" )
        
        f.write( "\n################### CLASS ASSIGNMENT #################\n" )
        for line in graph.graph( f"Spikes distribution (timestep={dt}):",
                                 [ (str( i ), x / np.sum( num_spikes_train ) * 100) for i, x in
                                   enumerate( num_spikes_class ) ] ):
            f.write( f"{line}\n" )
        f.write( f"\tTotal: {np.sum( num_spikes_class )}\n" )
        f.write( f"\tNormalised standard dev.: {np.std( num_spikes_class ) / np.mean( num_spikes_class )}\n" )
        f.write( "\nNeuron activations for each label:\n" )
        f.write( str( neuron_label_count ) + "\n" )
        f.write( "\nLabel associated to each neuron:\n" )
        f.write( str( neuron_label ) + "\n" )
        f.write( "\nNeuron set associated to each label:\n" )
        f.write( str( label_class ) + "\n" )
        f.write( f"Number of labels discovered: {len( label_class.keys() )}\n" )
        
        f.write( "\n###################### INFERENCE #####################\n" )
        for line in graph.graph( f"Spikes distribution (timestep={dt}):",
                                 [ (str( i ), x / np.sum( num_spikes_train ) * 100) for i, x in
                                   enumerate( num_spikes_test ) ] ):
            f.write( f"{line}\n" )
        f.write( f"\tTotal: {np.sum( num_spikes_test )}\n" )
        f.write( f"\tNormalised standard dev.: {np.std( num_spikes_test ) / np.mean( num_spikes_test )}\n" )
        f.write( "\nClassification results:\n" )
        f.write( f"\n\tAccuracy: {accuracy_score( test_labels.ravel(), prediction )}\n" )
        f.write( f"\tConfusion matrix:\n {confusion_matrix( test_labels.ravel(), prediction )}\n" )
    
    print( f"Saved data in {dir_data}" )
