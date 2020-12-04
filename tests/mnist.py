from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import nengo
from nengo.utils.filter_design import cont2discrete
import numpy as np
import tensorflow as tf

import nengo_dl
from nengo.utils.matplotlib import rasterplot

from memristor_nengo import extras

# set seed to ensure this example is reproducible
seed = 0
tf.random.set_seed( seed )
np.random.seed( seed )
rng = np.random.RandomState( seed )

# load mnist dataset
(train_images, train_labels), (test_images, test_labels,) = tf.keras.datasets.mnist.load_data()

# change inputs to -1--1 range
train_images = 2 * train_images / 255 - 1
test_images = 2 * test_images / 255 - 1

# reshape the labels to rank 3 (as expected in Nengo)
train_labels = train_labels[ :, None, None ]
test_labels = test_labels[ :, None, None ]

# plt.figure()
# plt.imshow( np.reshape( train_images[ 0 ], (28, 28) ), cmap="gray" )
# plt.axis( "off" )
# plt.title( str( train_labels[ 0, 0, 0 ] ) )
# plt.show()

train_images = np.array(
        [ x for i, x in enumerate( train_images ) if train_labels[ i ] == 5 or train_labels[ i ] == 7 ] )
test_images = np.array( [ x for i, x in enumerate( test_images ) if test_labels[ i ] == 5 or test_labels[ i ] == 7 ] )

presentation_time = 0.35

model = nengo.Network()
with model:
    inp = nengo.Node( nengo.processes.PresentInput( train_images, 0.35 ) )
    pre = nengo.Ensemble( n_neurons=784, dimensions=1 )
    post = nengo.Ensemble( n_neurons=2, dimensions=1 )
    
    nengo.Connection( inp, pre.neurons )
    
    conn = nengo.Connection( pre.neurons, post.neurons,
                             learning_rule_type=nengo.learning_rules.BCM(),
                             transform=np.random.random( (post.n_neurons, pre.n_neurons) )
                             )
    
    nengo.Connection( post.neurons, post.neurons,
                      transform=
                      (np.full( (post.n_neurons, post.n_neurons), 1 )
                       - np.eye( post.n_neurons )) * -2
                      )
    
    pre_probe = nengo.Probe( pre.neurons, sample_every=1000 )
    post_probe = nengo.Probe( post.neurons, sample_every=1000 )
    weight_probe = nengo.Probe( conn, "weights", sample_every=1000 )

with nengo_dl.Simulator( model ) as sim:
    sim.run( presentation_time * train_images.shape[ 0 ] )

fig, ax1 = plt.subplots()
rasterplot( sim.trange(), sim.data[ pre_probe ], ax1 )
ax1.set_ylabel( 'Neuron' )
ax1.set_xlabel( 'Time (s)' )
fig.get_axes()[ 0 ].annotate( "Pre" + " neural activity", (0.5, 0.94),
                              xycoords='figure fraction', ha='center',
                              fontsize=20
                              )
fig, ax1 = plt.subplots()
rasterplot( sim.trange(), sim.data[ post_probe ], ax1 )
ax1.set_ylabel( 'Neuron' )
ax1.set_xlabel( 'Time (s)' )
fig.get_axes()[ 0 ].annotate( "Post" + " neural activity", (0.5, 0.94),
                              xycoords='figure fraction', ha='center',
                              fontsize=20
                              )
fig, axes = plt.subplots( 2 )
for i, ax in enumerate( axes.flatten() ):
    ax.matshow( sim.data[ weight_probe ][ -1, i, ... ].reshape( (28, 28) ) )
    ax.set_title( f"Neuron {i + 1}" )
    ax.set_yticks( [ ] )
    ax.set_xticks( [ ] )
fig.suptitle( "Weights after learning" )

plt.show()
