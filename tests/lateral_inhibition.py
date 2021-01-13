from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import nengo
import numpy as np
import tensorflow as tf

import nengo_dl
from matplotlib.ticker import MultipleLocator
from nengo.utils.matplotlib import rasterplot
from memristor_nengo.extras import *
from memristor_nengo.neurons import *
from nengo import AdaptiveLIF

dir_name, dir_images, dir_data = make_timestamped_dir( root="../data/MNIST/" )

# set seed to ensure this example is reproducible
seed = 0
tf.random.set_seed( seed )
np.random.seed( seed )
rng = np.random.RandomState( seed )

# load mnist dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# change inputs to [0-1] range
train_images = train_images / 255
test_images = test_images / 255

# reshape the labels to rank 3 (as expected in Nengo)
train_labels = train_labels[ :, None, None ]
test_labels = test_labels[ :, None, None ]

# plt.figure()
# plt.imshow( np.reshape( train_images[ 0 ], (28, 28) ), cmap="gray" )
# plt.axis( "off" )
# plt.title( str( train_labels[ 0, 0, 0 ] ) )
# plt.show()

digits = (0, 1, 2)

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

train_images = train_images[ :10 ]
test_images = test_images[ :10 ]
train_labels = train_labels[ :10 ]
test_labels = test_labels[ :10 ]

print( train_labels.ravel() )

presentation_time = 0.35
pause_time = 0.15
sim_time = (presentation_time + pause_time) * train_images.shape[ 0 ]

model = nengo.Network()
with model:
    inp = nengo.Node( PresentInputWithPause( train_images, presentation_time, pause_time ) )
    pre = nengo.Ensemble( n_neurons=784, dimensions=1,
                          neuron_type=nengo.neurons.PoissonSpiking( nengo.LIFRate( amplitude=0.2 ) ),
                          encoders=nengo.dists.Choice( [ [ 1 ] ] ),
                          intercepts=nengo.dists.Choice( [ 0 ] ),
                          # max_rates=nengo.dists.Choice( [ 20, 22 ] )
                          seed=seed
                          )
    post = nengo.Ensemble( n_neurons=len( digits ) if digits else 10, dimensions=1,
                           neuron_type=AdaptiveLIFLateralInhibition( tau_inhibition=10 ),
                           # neuron_type=AdaptiveLIF(),
                           # neuron_type=LIF(),
                           encoders=nengo.dists.Choice( [ [ 1 ] ] ),
                           intercepts=nengo.dists.Choice( [ 0 ] ),
                           max_rates=nengo.dists.Choice( [ 20, 22 ] ),
                           seed=seed
                           )
    
    nengo.Connection( inp, pre.neurons )
    
    conn = nengo.Connection( pre.neurons, post.neurons,
                             learning_rule_type=nengo.learning_rules.Oja(),
                             transform=np.random.random( (post.n_neurons, pre.n_neurons) )
                             )
    
    # conn_rec = nengo.Connection( post.neurons, post.neurons,
    #                              transform=(np.full( (post.n_neurons, post.n_neurons), 1 ) - np.eye(
    #                                      post.n_neurons )) * -2,
    #                              # synapse=10 * dt
    #                              )
    
    pre_probe = nengo.Probe( pre.neurons, )
    post_probe = nengo.Probe( post.neurons, )
    weight_probe = nengo.Probe( conn, "weights", sample_every=sim_time )

print( "Pre:\n\t", pre.neuron_type )
print( "Post:\n\t", post.neuron_type )
print( "Rule:\n\t", conn.learning_rule_type )

with nengo.Simulator( model ) as sim:
    sim.run( sim_time )

major_ticks = np.arange( 0,
                         sim.trange()[ -1 ] + (presentation_time + pause_time),
                         (presentation_time + pause_time)
                         )[ 1: ]
minor_ticks = major_ticks - pause_time

fig1, ax = plt.subplots()
rasterplot( sim.trange(), sim.data[ pre_probe ], ax )
ax.set_ylabel( 'Neuron' )
ax.set_xlabel( 'Time (s)' )
fig1.get_axes()[ 0 ].annotate( "Pre" + " neural activity", (0.5, 0.94),
                               xycoords='figure fraction', ha='center',
                               fontsize=20
                               )
ax.xaxis.set_major_locator( MultipleLocator( presentation_time + pause_time ) )
for i, (mj, mn) in enumerate( zip( major_ticks, minor_ticks ) ):
    plt.axvline( x=mn, color='k', linestyle='--', alpha=0.5 )
    plt.axvline( x=mj, color='k', )
    ax.annotate( train_labels.ravel()[ i ], xy=(mn - 0.25, np.rint( pre.n_neurons / 2 )), xycoords='data' )
# fig1.show()

fig2, ax = plt.subplots()
rasterplot( sim.trange(), sim.data[ post_probe ], ax )
ax.set_ylabel( 'Neuron' )
ax.set_xlabel( 'Time (s)' )
fig2.get_axes()[ 0 ].annotate( "Post" + " neural activity", (0.5, 0.94),
                               xycoords='figure fraction', ha='center',
                               fontsize=20
                               )
ax.xaxis.set_major_locator( MultipleLocator( presentation_time + pause_time ) )
for i, (mj, mn) in enumerate( zip( major_ticks, minor_ticks ) ):
    plt.axvline( x=mn, color='k', linestyle='--', alpha=0.5 )
    plt.axvline( x=mj, color='k', )
    ax.annotate( train_labels.ravel()[ i ], xy=(mn - 0.25, np.rint( post.n_neurons / 2 )), xycoords='data' )
fig2.show()

fig3, axes = plt.subplots( 2 )
for i, ax in enumerate( axes.flatten() ):
    ax.matshow( sim.data[ weight_probe ][ -1, i, ... ].reshape( (28, 28) ) )
    ax.set_title( f"Neuron {i + 1}" )
    ax.set_yticks( [ ] )
    ax.set_xticks( [ ] )
fig3.suptitle( "Weights after learning" )
# fig3.show()

# fig1.savefig( dir_name + "pre" + ".eps" )
# fig2.savefig( dir_name + "post" + ".eps" )
# fig3.savefig( dir_name + "weights" + ".eps" )
