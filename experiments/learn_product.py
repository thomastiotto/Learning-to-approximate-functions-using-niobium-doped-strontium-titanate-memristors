import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import argparse
import os

os.environ[ "CUDA_DEVICE_ORDER" ] = "PCI_BUS_ID"

import nengo_dl
from nengo.processes import WhiteNoise
from nengo.dists import Gaussian
from nengo.learning_rules import PES

from memristor_nengo.extras import *
from memristor_nengo.learning_rules import mPES

parser = argparse.ArgumentParser()
parser.add_argument( "-T", "--sim_time", default=10, type=int )
parser.add_argument( "-d", "--device", default="/cpu:0" )
args = parser.parse_args()

sim_time = args.sim_time
learn_block_time = 2.5
device = args.device

learned_model = nengo.Network()
with learned_model:
    inp = nengo.Node(
            WhiteNoise( dist=Gaussian( 0, 0.05 ) ),
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
        
        if t % learn_block_time == 0:
            if out_inhibit == 0.0:
                out_inhibit = 2.0
            else:
                out_inhibit = 0.0
        
        return out_inhibit
    
    
    inhib = nengo.Node( cyclic_inhibit )
    nengo.Connection( inhib, error.neurons, transform=[ [ -1 ] ] * error.n_neurons )
    
    # -- probes
    product_probe_learn = nengo.Probe( product, synapse=0.01 )
    pre_probe_learn = nengo.Probe( pre, synapse=0.01 )
    post_probe_learn = nengo.Probe( post, synapse=0.01 )
    error_probe_learn = nengo.Probe( error, synapse=0.03 )

control_model = nengo.Network()
with control_model:
    inp = nengo.Node(
            WhiteNoise( dist=Gaussian( 0, 0.05 ) ),
            # WhiteSignal( 60, 5 ),
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
        
        if t % learn_block_time == 0:
            if out_inhibit == 0.0:
                out_inhibit = 2.0
            else:
                out_inhibit = 0.0
        
        return out_inhibit
    
    
    inhib = nengo.Node( cyclic_inhibit )
    nengo.Connection( inhib, error.neurons, transform=[ [ -1 ] ] * error.n_neurons )
    
    # -- probes
    product_probe_control = nengo.Probe( product, synapse=0.01 )
    pre_probe_control = nengo.Probe( pre, synapse=0.01 )
    post_probe_control = nengo.Probe( post, synapse=0.01 )
    error_probe_control = nengo.Probe( error, synapse=0.03 )

# 10 trail runs for each model
errors_iterations_learn = [ ]
errors_iterations_control = [ ]
for i in range( 10 ):
    print( "Iteration", i )
    with nengo_dl.Simulator( learned_model, device=device ) as learned_sim:
        learned_sim.run( sim_time )
    with nengo_dl.Simulator( control_model, device=device ) as control_sim:
        control_sim.run( sim_time )
    
    # essential statistics
    # split probe data into the trial run blocks
    product_data = np.array_split( learned_sim.data[ product_probe_learn ], sim_time / learn_block_time )
    post_data = np.array_split( learned_sim.data[ post_probe_learn ], sim_time / learn_block_time )
    # extract learning blocks
    learn_product_data = np.array( [ x for i, x in enumerate( product_data ) if i % 2 == 0 ] )
    test_product_data = np.array( [ x for i, x in enumerate( product_data ) if i % 2 != 0 ] )
    # extract testing blocks
    learn_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 == 0 ] )
    test_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 != 0 ] )
    
    # compute testing error for learn network
    total_error = np.sum( np.abs( test_post_data - test_product_data ), axis=1 )
    errors_iterations_learn.append( total_error )
    
    # split probe data into the trial run blocks
    product_data = np.array_split( control_sim.data[ product_probe_control ], sim_time / learn_block_time )
    post_data = np.array_split( control_sim.data[ post_probe_control ], sim_time / learn_block_time )
    # extract learning blocks
    learn_product_data = np.array( [ x for i, x in enumerate( product_data ) if i % 2 == 0 ] )
    test_product_data = np.array( [ x for i, x in enumerate( product_data ) if i % 2 != 0 ] )
    # extract testing blocks
    learn_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 == 0 ] )
    test_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 != 0 ] )
    
    # compute testing error for control network
    total_error = np.sum( np.abs( test_post_data - test_product_data ), axis=1 )
    errors_iterations_control.append( total_error )

# compute mean testing error and confidence intervals
errors_mean_learn = np.mean( errors_iterations_learn, axis=0 )
errors_mean_control = np.mean( errors_iterations_control, axis=0 )
ci_learn = st.t.interval( 0.95, len( errors_iterations_learn ) - 1, loc=np.mean( errors_iterations_learn ),
                          scale=st.sem( errors_iterations_learn ) )
ci_control = st.t.interval( 0.95, len( errors_iterations_control ) - 1, loc=np.mean( errors_iterations_control ),
                            scale=st.sem( errors_iterations_control ) )

# plot testing error
fig, ax = plt.subplots()
x = range( errors_mean_learn.shape[ 0 ] )
ax.plot( x, errors_mean_learn, label="Learned (mPES)" )
ax.plot( x, ci_learn[ 0 ], linestyle="--", alpha=0.5 )
ax.plot( x, ci_learn[ 1 ], linestyle="--", alpha=0.5 )
ax.plot( x, errors_mean_learn, label="Learned (mPES)" )
ax.plot( x, ci_control[ 0 ], linestyle="--", alpha=0.5 )
ax.plot( x, ci_control[ 1 ], linestyle="--", alpha=0.5 )
ax.plot( x, errors_mean_control, label="Control (PES)" )
ax.legend( loc="best" )
fig.show()
