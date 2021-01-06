import matplotlib.pyplot as plt
from random import randrange
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random

import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

from matplotlib.ticker import MultipleLocator
from nengo.utils.matplotlib import rasterplot
from memristor_nengo.extras import *
from memristor_nengo.neurons import *

dir_name, dir_images, dir_data = make_timestamped_dir( root="../data/MNIST/" )

# load mnist dataset
(train_images, train_labels), (test_images, test_labels,) = tf.keras.datasets.mnist.load_data()

# change inputs to [0-1] range
train_images = train_images / 255
test_images = test_images / 255

# reshape the labels to rank 3 (as expected in Nengo)
train_labels = train_labels[ :, None, None ]
test_labels = test_labels[ :, None, None ]

digits = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
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

num_samples = 10000
if num_samples:
    train_images = train_images[ :num_samples ]
    test_images = test_images[ :num_samples ]
    train_labels = train_labels[ :num_samples ]
    test_labels = test_labels[ :num_samples ]

print( "######################################################",
       "###################### DEFINITION ####################",
       "######################################################",
       sep="\n" )

seed = None
random.seed = seed
presentation_time = 0.35
pause_time = 0.15
sim_time = (presentation_time + pause_time) * train_images.shape[ 0 ]
dt = 0.001
make_video = True
sample_every = 100 * dt
sample_every_weights = num_samples * dt if make_video else sim_time
post_n_neurons = 10

model = nengo.Network()
with model:
    inp = nengo.Node( PresentInputWithPause( train_images, presentation_time, pause_time ) )
    pre = nengo.Ensemble( n_neurons=784, dimensions=1,
                          neuron_type=nengo.neurons.PoissonSpiking( nengo.LIFRate() ),
                          encoders=nengo.dists.Choice( [ [ 1 ] ] ),
                          intercepts=nengo.dists.Choice( [ 0 ] ),
                          # max_rates=nengo.dists.Choice( [ 20, 22 ] )
                          seed=seed
                          )
    post = nengo.Ensemble( n_neurons=post_n_neurons, dimensions=1,
                           neuron_type=AdaptiveLIFLateralInhibition( inhibition=10 ),
                           # neuron_type=AdaptiveLIF(),
                           # neuron_type=LIF(),
                           encoders=nengo.dists.Choice( [ [ 1 ] ] ),
                           intercepts=nengo.dists.Choice( [ 0 ] ),
                           max_rates=nengo.dists.Choice( [ 20, 22 ] ),
                           seed=seed
                           )
    
    nengo.Connection( inp, pre.neurons )
    
    conn = nengo.Connection( pre.neurons, post.neurons,
                             learning_rule_type=nengo.learning_rules.Oja( beta=2.5 ),
                             transform=np.random.random( (post.n_neurons, pre.n_neurons) )
                             )
    
    pre_probe = nengo.Probe( pre.neurons, sample_every=sample_every )
    post_probe = nengo.Probe( post.neurons, sample_every=sample_every )
    weight_probe = nengo.Probe( conn, "weights", sample_every=sample_every_weights )
    adaptation_probe = nengo.Probe( post.neurons, "adaptation", sample_every=sim_time )

un_train, cnt_train = np.unique( train_labels, return_counts=True )
un_test, cnt_test = np.unique( test_labels, return_counts=True )

with open( dir_data + "results.txt", "w" ) as f:
    f.write( "\n###################### DEFINITION ####################\n" )
    f.write( "\nDigits distribution:\n" )
    f.write( "\tTrain:\n" )
    
    for x, c in zip( un_train, cnt_train ):
        f.write( f"\t\t{x}: {c}"
                 f"[{c / len( train_labels ) * 100} %]\n" )
    f.write( "\tTest:\n" )
    for x, c in zip( un_test, cnt_test ):
        f.write( f"\t\t{x}: {c}"
                 f"[{c / len( test_labels ) * 100} %]\n" )
    f.write( f"Pre:\n\t {pre.neuron_type} \n\tNeurons: {pre.n_neurons} \n\tRate: {pre.max_rates}\n" )
    f.write( f"Post:\n\t {post.neuron_type} \n\tNeurons: {post.n_neurons} \n\tRate: {post.max_rates}\n" )
    f.write( f"Rule:\n\t {conn.learning_rule_type}\n" )

print( "Digits distribution:" )
print( "\tTrain:" )
for x, c in zip( un_train, cnt_train ):
    print( f"\t\t{x}: {c}"
           f"[{c / len( train_labels ) * 100} %]" )
print( "\tTest:" )
for x, c in zip( un_test, cnt_test ):
    print( f"\t\t{x}: {c}"
           f"[{c / len( test_labels ) * 100} %]" )
print( "Pre:\n\t", pre.neuron_type, "\n\tNeurons:", pre.n_neurons, "\n\tRate:", pre.max_rates )
print( "Post:\n\t", post.neuron_type, "\n\tNeurons:", post.n_neurons, "\n\tRate:", post.max_rates )
print( "Rule:\n\t", conn.learning_rule_type )

print( "######################################################",
       "####################### TRAINING #####################",
       "######################################################",
       sep="\n" )

with nengo.Simulator( model ) as sim_train:
    sim_train.run( sim_time )

# print number of recorded spikes
num_spikes_train = np.sum( sim_train.data[ post_probe ] > 0, axis=0 )
print( f"Number of spikes (timestep={sample_every}):" )
for i, x in enumerate( num_spikes_train ):
    print( f"\tNeuron {i}:", x, f"[{x / np.sum( num_spikes_train ) * 100} %]" )
print( "\tTotal:", np.sum( num_spikes_train ) )

fig1, ax = plt.subplots()
rasterplot( sim_train.trange( sample_every=sample_every ), sim_train.data[ pre_probe ], ax )
ax.set_ylabel( 'Neuron' )
ax.set_xlabel( 'Time (s)' )
fig1.get_axes()[ 0 ].annotate( "Pre" + " neural activity", (0.5, 0.94),
                               xycoords='figure fraction', ha='center',
                               fontsize=20
                               )
# fig1.show()

fig2, ax = plt.subplots()
rasterplot( sim_train.trange( sample_every=sample_every ), sim_train.data[ post_probe ], ax )
ax.set_ylabel( 'Neuron' )
ax.set_xlabel( 'Time (s)' )
fig2.get_axes()[ 0 ].annotate( "Post" + " neural activity", (0.5, 0.94),
                               xycoords='figure fraction', ha='center',
                               fontsize=20
                               )
fig2.show()

fig3, axes = plt.subplots( int( post.n_neurons / 2 ), int( post.n_neurons / int( post.n_neurons / 2 ) ) )
for i, ax in enumerate( axes.flatten() ):
    ax.matshow( sim_train.data[ weight_probe ][ -1, i, ... ].reshape( (28, 28) ) )
    ax.set_title( f"Neuron {i}" )
    ax.set_yticks( [ ] )
    ax.set_xticks( [ ] )
fig3.suptitle( "Weights after learning" )
fig3.tight_layout()
fig3.show()

if make_video:
    for i in range( post_n_neurons ):
        generate_heatmap( sim_train.data[ weight_probe ], dir_images, neuron=i )

print( "######################################################",
       "################### CLASS ASSIGNMENT #################",
       "######################################################",
       sep="\n" )
# load weights found during training and freeze them
conn.transform = sim_train.data[ weight_probe ][ -1 ].squeeze()
conn.learning_rule_type = nengo.learning_rules.Oja( learning_rate=0 )
# load last neuron thresholds from training and freeze them
post.neuron_type = AdaptiveLIFLateralInhibition( inhibition=10,
                                                 tau_n=1e20,
                                                 inc_n=0,
                                                 initial_state={
                                                         "adaptation": sim_train.data[ adaptation_probe ].squeeze() } )

# set post probe to record every spike for statistics
post_probe.sample_every = dt

with nengo.Simulator( model ) as sim_class:
    sim_class.run( sim_time )

# print number of recorded spikes
num_spikes_class = np.sum( sim_class.data[ post_probe ] > 0, axis=0 )
print( f"Number of spikes (timestep={dt}):" )
for i, x in enumerate( num_spikes_class ):
    print( f"\tNeuron {i}:", x, f"[{x / np.sum( num_spikes_class ) * 100} %]" )
print( "\tTotal:", np.sum( num_spikes_class ) )

# remove spikes that happened during pause period
mask = np.concatenate( (np.ones( int( presentation_time / dt ) + 1 ), np.zeros( int( pause_time / dt ) )) ) \
    .astype( bool )
clean_post_spikes_class = sim_class.data[ post_probe ].reshape( num_samples, -1, post_n_neurons )[ :, mask, ... ]

# count neuron activations in response to each example
neuron_activations_class = np.count_nonzero( clean_post_spikes_class, axis=1 )
neuron_label_count = { neur: { lab: 0 for lab in un_train } for neur in range( post_n_neurons ) }

# count how many times each neuron spiked for each label across examples
for t, lab in enumerate( train_labels.ravel() ):
    for neur in range( post_n_neurons ):
        neuron_label_count[ neur ][ lab ] += neuron_activations_class[ t, neur ]
print( "Neuron activations for each label:\n", end="" )
pprint_dict( neuron_label_count, level=1 )

# associate each neuron with the label it spiked most for
# if the neuron never spiked pick a random label
neuron_label = { neur: max( lab, key=lab.get ) if all( v != 0 for v in lab.values() ) else randrange(
        len( un_train ) ) for neur, lab in neuron_label_count.items() }
print( "Label associated to each neuron:\n", end="" )
pprint_dict( neuron_label, level=1 )

# group neurons into classes based on their label
label_class = { }
for neur, lab in neuron_label.items():
    label_class.setdefault( lab, list() ).append( neur )
print( "Neuron set associated to each label:\n", end="" )
pprint_dict( label_class, level=1 )

print( "######################################################",
       "###################### INFERENCE #####################",
       "######################################################",
       sep="\n" )

# switch to test set
inp.output = PresentInputWithPause( test_images, presentation_time, pause_time )

with nengo.Simulator( model ) as sim_test:
    sim_test.run( sim_time )

# print number of recorded spikes
num_spikes_test = np.sum( sim_test.data[ post_probe ] > 0, axis=0 )
print( f"Number of spikes (timestep={dt}):" )
for i, x in enumerate( num_spikes_test ):
    print( f"\tNeuron {i}:", x, f"[{x / np.sum( num_spikes_test ) * 100} %]" )
print( "\tTotal:", np.sum( num_spikes_test ) )

# remove spikes that happened during pause period
clean_post_spikes_test = sim_test.data[ post_probe ].reshape( num_samples, -1, post_n_neurons )[ :, mask, ... ]

# count neuron activations in response to each example
neuron_activations_test = np.count_nonzero( clean_post_spikes_test, axis=1 )

# use the neuron class with highest average activation as class prediction at each timestep
prediction = [ ]
for t in range( num_samples ):
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

fig1.savefig( dir_name + "pre" + ".eps" )
fig2.savefig( dir_name + "post" + ".eps" )
fig3.savefig( dir_name + "weights" + ".eps" )
print( f"Saved plots in {dir_images}" )

with open( dir_data + "results.txt", "a" ) as f:
    f.write( "\n####################### TRAINING #####################\n" )
    f.write( f"\nNumber of spikes (timestep={sample_every}):\n" )
    for i, x in enumerate( num_spikes_train ):
        f.write( f"\tNeuron {i}: {x} [{x / np.sum( num_spikes_train ) * 100} %]\n" )
    f.write( f"\tTotal: {np.sum( num_spikes_train )}\n" )
    
    f.write( "\n################### CLASS ASSIGNMENT #################\n" )
    f.write( f"\nNumber of spikes (timestep={dt}):\n" )
    for i, x in enumerate( num_spikes_class ):
        f.write( f"\tNeuron {i}: {x} [{x / np.sum( num_spikes_class ) * 100} %]\n" )
    f.write( f"\tTotal: {np.sum( num_spikes_class )}\n" )
    f.write( "\nNeuron activations for each label:\n" )
    f.write( str( neuron_label_count ) + "\n" )
    f.write( "\nLabel associated to each neuron:\n" )
    f.write( str( neuron_label ) + "\n" )
    f.write( "\nNeuron set associated to each label:\n" )
    f.write( str( label_class ) + "\n" )
    
    f.write( "\n###################### INFERENCE #####################\n" )
    f.write( f"\nNumber of spikes (timestep={dt}):\n" )
    for i, x in enumerate( num_spikes_test ):
        f.write( f"\tNeuron {i}: {x} [{x / np.sum( num_spikes_test ) * 100} %]\n" )
    f.write( f"\tTotal: {np.sum( num_spikes_test )}\n" )
    f.write( "\nClassification results:\n" )
    f.write( f"\n\tAccuracy: {accuracy_score( test_labels.ravel(), prediction )}\n" )
    f.write( f"\tConfusion matrix:\n {confusion_matrix( test_labels.ravel(), prediction )}\n" )

print( f"Saved data in {dir_data}" )
