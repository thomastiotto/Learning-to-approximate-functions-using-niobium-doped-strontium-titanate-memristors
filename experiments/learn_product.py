import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import argparse
import os
import sys

import nengo_dl
from nengo.processes import WhiteNoise, WhiteSignal
from nengo.dists import Gaussian
from nengo.learning_rules import PES

from memristor_nengo.extras import *
from memristor_nengo.learning_rules import mPES

setup()

parser = argparse.ArgumentParser()
parser.add_argument( "-T", "--sim_time", default=50, type=float )
parser.add_argument( "-I", "--iterations", default=10, type=int )
parser.add_argument( "-d", "--device", default="/cpu:0" )
args = parser.parse_args()

sim_time = args.sim_time
iterations = args.iterations
learn_block_time = 2.5
device = args.device
directory = "../data/"

dir_name, dir_images, dir_data = make_timestamped_dir(
        root=directory + "trevor/" + "product/" )
print( "Reserved folder", dir_name )

learned_model = nengo.Network()
with learned_model:
    inp = nengo.Node(
            # WhiteNoise( dist=Gaussian( 0, 0.05 ) ),
            WhiteSignal( 60, high=5 ),
            size_out=2
            )
    pre = nengo.Ensemble( 200, dimensions=2 )
    product = nengo.Ensemble( 100, dimensions=1 )
    post = nengo.Ensemble( 200, dimensions=1 )
    error = nengo.Ensemble( 100, dimensions=1 )
    
    nengo.Connection( inp, pre )
    nengo.Connection( inp, product, function=lambda x: x[ 0 ] * x[ 1 ] )
    nengo.Connection( post, error )
    nengo.Connection( product, error, transform=-1 )
    
    # -- learning connection
    conn = nengo.Connection(
            pre.neurons,
            post.neurons,
            transform=np.zeros( (post.n_neurons, pre.n_neurons) ),
            learning_rule_type=mPES( gain=1e3 ),
            )
    
    nengo.Connection( error, conn.learning_rule )
    
    # -- inhibit error every 2.5 seconds
    out_inhibit = 0.0
    
    
    def cyclic_inhibit( t ):
        global out_inhibit
        
        if t % learn_block_time == 0 and t != 0:
            if out_inhibit == 0.0:
                out_inhibit = 2.0
            else:
                out_inhibit = 0.0
        
        return out_inhibit
    
    
    inhib = nengo.Node( cyclic_inhibit )
    nengo.Connection( inhib, error.neurons, transform=[ [ -1 ] ] * error.n_neurons )
    
    # -- probes
    product_probe_mpes = nengo.Probe( product, synapse=0.01 )
    pre_probe_mpes = nengo.Probe( pre, synapse=0.01 )
    post_probe_mpes = nengo.Probe( post, synapse=0.01 )
    error_probe_mpes = nengo.Probe( error, synapse=0.03 )

control_model_pes = nengo.Network()
with control_model_pes:
    inp = nengo.Node(
            # WhiteNoise( dist=Gaussian( 0, 0.05 ) ),
            WhiteSignal( 60, high=5 ),
            size_out=2
            )
    pre = nengo.Ensemble( 200, dimensions=2 )
    product = nengo.Ensemble( 100, dimensions=1 )
    post = nengo.Ensemble( 200, dimensions=1 )
    error = nengo.Ensemble( 100, dimensions=1 )
    
    nengo.Connection( inp, pre )
    nengo.Connection( inp, product, function=lambda x: x[ 0 ] * x[ 1 ] )
    nengo.Connection( post, error )
    nengo.Connection( product, error, transform=-1 )
    
    # -- learning connection
    conn = nengo.Connection(
            pre,
            post,
            function=lambda x: np.random.random( 1 ),
            learning_rule_type=PES(),
            )
    
    nengo.Connection( error, conn.learning_rule )
    
    # -- inhibit error every 2.5 seconds
    out_inhibit = 0.0
    
    
    def cyclic_inhibit( t ):
        global out_inhibit
        
        if t % learn_block_time == 0 and t != 0:
            if out_inhibit == 0.0:
                out_inhibit = 2.0
            else:
                out_inhibit = 0.0
        
        return out_inhibit
    
    
    inhib = nengo.Node( cyclic_inhibit )
    nengo.Connection( inhib, error.neurons, transform=[ [ -1 ] ] * error.n_neurons )
    
    # -- probes
    product_probe_pes = nengo.Probe( product, synapse=0.01 )
    pre_probe_pes = nengo.Probe( pre, synapse=0.01 )
    post_probe_pes = nengo.Probe( post, synapse=0.01 )
    error_probe_pes = nengo.Probe( error, synapse=0.03 )

control_model_nef = nengo.Network()
with control_model_nef:
    inp = nengo.Node(
            # WhiteNoise( dist=Gaussian( 0, 0.05 ) ),
            WhiteSignal( 60, high=5 ),
            size_out=2
            )
    pre = nengo.Ensemble( 200, dimensions=2 )
    product = nengo.Ensemble( 100, dimensions=1 )
    post = nengo.Ensemble( 200, dimensions=1 )
    error = nengo.Ensemble( 100, dimensions=1 )
    
    nengo.Connection( inp, pre )
    nengo.Connection( inp, product, function=lambda x: x[ 0 ] * x[ 1 ] )
    nengo.Connection( post, error )
    nengo.Connection( product, error, transform=-1 )
    
    # -- learning connection
    conn = nengo.Connection(
            pre,
            post,
            function=lambda x: x[ 0 ] * x[ 1 ]
            )
    
    # -- inhibit error every 2.5 seconds
    out_inhibit = 0.0
    
    
    def cyclic_inhibit( t ):
        global out_inhibit
        
        if t % learn_block_time == 0 and t != 0:
            if out_inhibit == 0.0:
                out_inhibit = 2.0
            else:
                out_inhibit = 0.0
        
        return out_inhibit
    
    
    inhib = nengo.Node( cyclic_inhibit )
    nengo.Connection( inhib, error.neurons, transform=[ [ -1 ] ] * error.n_neurons )
    
    # -- probes
    product_probe_nef = nengo.Probe( product, synapse=0.01 )
    pre_probe_nef = nengo.Probe( pre, synapse=0.01 )
    post_probe_nef = nengo.Probe( post, synapse=0.01 )
    error_probe_nef = nengo.Probe( error, synapse=0.03 )

# trail runs for each model
errors_iterations_mpes = [ ]
errors_iterations_pes = [ ]
errors_iterations_nef = [ ]
for i in range( iterations ):
    print( "Iteration", i )
    with nengo_dl.Simulator( learned_model, device=device ) as sim_mpes:
        print( "Learning network (mPES)" )
        sim_mpes.run( sim_time )
    with nengo_dl.Simulator( control_model_pes, device=device ) as sim_pes:
        print( "Control network (PES)" )
        sim_pes.run( sim_time )
    with nengo_dl.Simulator( control_model_nef, device=device ) as sim_nef:
        print( "Control network (NEF)" )
        sim_nef.run( sim_time )
    
    # essential statistics
    
    # mPES
    # split probe data into the trial run blocks
    product_data = np.array_split( sim_mpes.data[ product_probe_mpes ], sim_time / learn_block_time )
    post_data = np.array_split( sim_mpes.data[ post_probe_mpes ], sim_time / learn_block_time )
    # extract learning blocks
    learn_product_data = np.array( [ x for i, x in enumerate( product_data ) if i % 2 == 0 ] )
    test_product_data = np.array( [ x for i, x in enumerate( product_data ) if i % 2 != 0 ] )
    # extract testing blocks
    learn_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 == 0 ] )
    test_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 != 0 ] )
    
    # compute testing error for learn network
    total_error = np.sum( np.abs( test_post_data - test_product_data ), axis=1 )
    errors_iterations_mpes.append( total_error )
    
    # PES
    # split probe data into the trial run blocks
    product_data = np.array_split( sim_pes.data[ product_probe_pes ], sim_time / learn_block_time )
    post_data = np.array_split( sim_pes.data[ post_probe_pes ], sim_time / learn_block_time )
    # extract learning blocks
    learn_product_data = np.array( [ x for i, x in enumerate( product_data ) if i % 2 == 0 ] )
    test_product_data = np.array( [ x for i, x in enumerate( product_data ) if i % 2 != 0 ] )
    # extract testing blocks
    learn_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 == 0 ] )
    test_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 != 0 ] )
    
    # compute testing error for control network
    total_error = np.sum( np.abs( test_post_data - test_product_data ), axis=1 )
    errors_iterations_pes.append( total_error )
    
    # NEF
    # split probe data into the trial run blocks
    product_data = np.array_split( sim_nef.data[ product_probe_nef ], sim_time / learn_block_time )
    post_data = np.array_split( sim_nef.data[ post_probe_nef ], sim_time / learn_block_time )
    # extract learning blocks
    learn_product_data = np.array( [ x for i, x in enumerate( product_data ) if i % 2 == 0 ] )
    test_product_data = np.array( [ x for i, x in enumerate( product_data ) if i % 2 != 0 ] )
    # extract testing blocks
    learn_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 == 0 ] )
    test_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 != 0 ] )
    
    # compute testing error for control network
    total_error = np.sum( np.abs( test_post_data - test_product_data ), axis=1 )
    errors_iterations_nef.append( total_error )

# compute mean testing error and confidence intervals
errors_mean_mpes = np.mean( errors_iterations_mpes, axis=0 )
errors_mean_pes = np.mean( errors_iterations_pes, axis=0 )
errors_mean_nef = np.mean( errors_iterations_nef, axis=0 )


# 95% confidence interval
def ci( data ):
    return \
        np.mean( data, axis=0 ), \
        np.mean( data, axis=0 ) + 1.960 * np.std( data, axis=0 ) / np.sqrt( len( data ) ), \
        np.mean( data, axis=0 ) - 1.960 * np.std( data, axis=0 ) / np.sqrt( len( data ) )


ci_mpes = ci( errors_iterations_mpes )
ci_pes = ci( errors_iterations_pes )
ci_nef = ci( errors_iterations_nef )

# plot testing error
fig, ax = plt.subplots()
x = range( errors_mean_mpes.shape[ 0 ] )
plt.xticks( x, np.array( x ) * sim_time / errors_mean_mpes.shape[ 0 ] + 2 * learn_block_time )
ax.plot( x, ci_mpes[ 0 ], label="Learned (mPES)", c="b" )
ax.plot( x, ci_mpes[ 1 ], linestyle="--", alpha=0.5, c="b" )
ax.plot( x, ci_mpes[ 2 ], linestyle="--", alpha=0.5, c="b" )
ax.plot( x, ci_pes[ 0 ], label="Control (PES)", c="g" )
ax.plot( x, ci_pes[ 1 ], linestyle="--", alpha=0.5, c="g" )
ax.plot( x, ci_pes[ 2 ], linestyle="--", alpha=0.5, c="g" )
ax.plot( x, ci_nef[ 0 ], label="Control (NEF)", c="r" )
ax.plot( x, ci_nef[ 1 ], linestyle="--", alpha=0.5, c="r" )
ax.plot( x, ci_nef[ 2 ], linestyle="--", alpha=0.5, c="r" )
ax.legend( loc="best" )
fig.show()

# noinspection PyTypeChecker
np.savetxt( dir_data + "results.csv",
            np.squeeze(
                    np.stack(
                            (errors_mean_mpes, ci_mpes[ 0 ], ci_mpes[ 1 ],
                             errors_mean_pes, ci_pes[ 0 ], ci_pes[ 1 ],
                             errors_mean_nef, ci_nef[ 0 ], ci_nef[ 1 ],
                             ),
                            axis=1
                            )
                    ),
            delimiter=",",
            header="Mean mPES error,CI mPES +,CI mPES -,"
                   "Mean PES error,CI PES +,CI PES -,"
                   "Mean NEF error,CI NEF +,CI NEF -,",
            comments="" )
print( f"Saved results in {dir_data}" )
fig.savefig( dir_images + "product" + ".pdf" )
print( f"Saved plots in {dir_images}" )
