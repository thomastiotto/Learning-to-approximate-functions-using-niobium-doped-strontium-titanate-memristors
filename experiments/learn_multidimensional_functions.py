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
parser.add_argument( "-E", "--experiment", choices=[ 1, 2, 3, 4, 5 ], type=int,
                     help="1: Product 2: Combined product" )
parser.add_argument( "-T", "--sim_time", default=None, type=float )
parser.add_argument( "-I", "--iterations", default=10, type=int )
parser.add_argument( "-g", "--gain", default=1e3, type=float )
parser.add_argument( "-d", "--device", default="/cpu:0" )
args = parser.parse_args()

experiment = args.experiment
if experiment == 1:
    exp_string = "PRODUCT experiment"
    exp_name = "Multiplying two numbers"
    function_to_learn = lambda x: x[ 0 ] * x[ 1 ]
    # [ pre, post, ground_truth, error ]
    neurons = [ 200, 200, 100, 100 ]
    dimensions = [ 2, 1, 1, 1 ]
    sim_time = 50
if experiment == 2:
    exp_string = "COMBINED PRODUCTS experiment"
    exp_name = "Combining two products"
    function_to_learn = lambda x: x[ 0 ] * x[ 1 ] + x[ 2 ] * x[ 3 ]
    # [ pre, post, ground_truth, error ]
    neurons = [ 400, 400, 100, 100 ]
    dimensions = [ 4, 1, 1, 1 ]
    sim_time = 100
if experiment == 3:
    exp_string = "SEPARATE PRODUCTS experiment"
    exp_name = "Three separate products"
    function_to_learn = lambda x: [ x[ 0 ] * x[ 1 ], x[ 0 ] * x[ 2 ], x[ 1 ] * x[ 2 ] ]
    # [ pre, post, ground_truth, error ]
    neurons = [ 300, 300, 300, 300 ]
    dimensions = [ 3, 3, 3, 3 ]
    sim_time = 100
if experiment == 4:
    exp_string = "2D CIRCULAR CONVOLUTIONS experiment"
    exp_name = "Two-dimensional circular convolution"
    # [ pre, post, ground_truth, error,conv ]
    neurons = [ 400, 400, 200, 200, 200 ]
    dimensions = [ 4, 2, 2, 2, 2 ]
    function_to_learn = lambda x: np.fft.ifft(
            np.fft.fft( x[ :int( dimensions[ 0 ] / 2 ) ] ) * np.fft.fft( x[ int( dimensions[ 0 ] / 2 ): ] )
            )
    sim_time = 200
if experiment == 5:
    exp_string = "3D CIRCULAR CONVOLUTIONS experiment"
    exp_name = "Two-dimensional circular convolution"
    # [ pre, post, ground_truth, error,conv ]
    neurons = [ 600, 300, 300, 300, 300 ]
    dimensions = [ 6, 3, 3, 3, 3 ]
    function_to_learn = lambda x: np.fft.ifft(
            np.fft.fft( x[ :int( dimensions[ 0 ] / 2 ) ] ) * np.fft.fft( x[ int( dimensions[ 0 ] / 2 ): ] )
            )
    sim_time = 400

assert 'exp_name' in locals()

if args.sim_time is not None:
    sim_time = args.sim_time
iterations = args.iterations
gain = args.gain
learn_block_time = 2.5
device = args.device
directory = "../data/"

print( exp_string )
dir_name, dir_images, dir_data = make_timestamped_dir(
        root=directory + "trevor/" + exp_name + "/" )
print( "Reserved folder", dir_name )


def LearningModel( neurons, dimensions, learning_rule, function_to_learn ):
    with nengo.Network() as function_learning_model:
        function_learning_model.inp = nengo.Node(
                # WhiteNoise( dist=Gaussian( 0, 0.05 ) ),
                WhiteSignal( 60, high=5 ),
                size_out=dimensions[ 0 ]
                )
        function_learning_model.pre = nengo.Ensemble( neurons[ 0 ], dimensions=dimensions[ 0 ] )
        function_learning_model.post = nengo.Ensemble( neurons[ 1 ], dimensions=dimensions[ 1 ] )
        function_learning_model.ground_truth = nengo.Ensemble( neurons[ 2 ], dimensions=dimensions[ 2 ] )
        function_learning_model.error = nengo.Ensemble( neurons[ 3 ], dimensions=dimensions[ 3 ] )
        
        nengo.Connection( function_learning_model.inp, function_learning_model.pre )
        nengo.Connection( function_learning_model.inp, function_learning_model.ground_truth,
                          function=function_to_learn )
        nengo.Connection( function_learning_model.post, function_learning_model.error )
        nengo.Connection( function_learning_model.ground_truth, function_learning_model.error, transform=-1 )
        
        if learning_rule:
            # -- learning connection
            function_learning_model.conn = nengo.Connection(
                    function_learning_model.pre.neurons,
                    function_learning_model.post.neurons,
                    transform=np.zeros(
                            (function_learning_model.post.n_neurons, function_learning_model.pre.n_neurons) ),
                    learning_rule_type=learning_rule
                    )
            nengo.Connection( function_learning_model.error, function_learning_model.conn.learning_rule )
        else:
            function_learning_model.conn = nengo.Connection(
                    function_learning_model.pre,
                    function_learning_model.post,
                    function=function_to_learn
                    )
        
        class cyclic_inhibit:
            def __init__( self, cycle_time ):
                self.out_inhibit = 0.0
                self.cycle_time = cycle_time
            
            def step( self, t ):
                if t % self.cycle_time == 0 and t != 0:
                    if self.out_inhibit == 0.0:
                        self.out_inhibit = 2.0
                    else:
                        self.out_inhibit = 0.0
                
                return self.out_inhibit
        
        function_learning_model.inhib = nengo.Node( cyclic_inhibit( learn_block_time ).step )
        nengo.Connection( function_learning_model.inhib, function_learning_model.error.neurons,
                          transform=[ [ -1 ] ] * function_learning_model.error.n_neurons )
        
        # -- probes
        function_learning_model.ground_truth_probe = nengo.Probe( function_learning_model.ground_truth, synapse=0.01 )
        function_learning_model.pre_probe = nengo.Probe( function_learning_model.pre, synapse=0.01 )
        function_learning_model.post_probe = nengo.Probe( function_learning_model.post, synapse=0.01 )
        function_learning_model.error_probe = nengo.Probe( function_learning_model.error, synapse=0.03 )
    
    return function_learning_model


def LearningConvolutionModel( neurons, dimensions, learning_rule, function_to_learn ):
    with nengo.Network() as function_learning_model:
        function_learning_model.inp = nengo.Node(
                # WhiteNoise( dist=Gaussian( 0, 0.05 ) ),
                WhiteSignal( 60, high=5 ),
                size_out=dimensions[ 0 ]
                )
        function_learning_model.pre = nengo.Ensemble( neurons[ 0 ], dimensions=dimensions[ 0 ] )
        function_learning_model.post = nengo.Ensemble( neurons[ 1 ], dimensions=dimensions[ 1 ] )
        function_learning_model.ground_truth = nengo.Ensemble( neurons[ 2 ], dimensions=dimensions[ 2 ] )
        function_learning_model.error = nengo.Ensemble( neurons[ 3 ], dimensions=dimensions[ 3 ] )
        function_learning_model.conv = nengo.networks.CircularConvolution( neurons[ 4 ], dimensions[ 4 ] )
        
        nengo.Connection( function_learning_model.inp, function_learning_model.pre )
        nengo.Connection( function_learning_model.inp[ :int( dimensions[ 0 ] / 2 ) ],
                          function_learning_model.conv.input_a )
        nengo.Connection( function_learning_model.inp[ int( dimensions[ 0 ] / 2 ): ],
                          function_learning_model.conv.input_b )
        nengo.Connection( function_learning_model.conv.output, function_learning_model.ground_truth )
        nengo.Connection( function_learning_model.post, function_learning_model.error )
        nengo.Connection( function_learning_model.ground_truth, function_learning_model.error, transform=-1 )
        
        if learning_rule:
            # -- learning connection
            function_learning_model.conn = nengo.Connection(
                    function_learning_model.pre.neurons,
                    function_learning_model.post.neurons,
                    transform=np.zeros(
                            (function_learning_model.post.n_neurons, function_learning_model.pre.n_neurons) ),
                    learning_rule_type=learning_rule
                    )
            nengo.Connection( function_learning_model.error, function_learning_model.conn.learning_rule )
        else:
            function_learning_model.conn = nengo.Connection(
                    function_learning_model.pre,
                    function_learning_model.post,
                    function=function_to_learn
                    )
        
        class cyclic_inhibit:
            def __init__( self, cycle_time ):
                self.out_inhibit = 0.0
                self.cycle_time = cycle_time
            
            def step( self, t ):
                if t % self.cycle_time == 0 and t != 0:
                    if self.out_inhibit == 0.0:
                        self.out_inhibit = 2.0
                    else:
                        self.out_inhibit = 0.0
                
                return self.out_inhibit
        
        function_learning_model.inhib = nengo.Node( cyclic_inhibit( learn_block_time ).step )
        nengo.Connection( function_learning_model.inhib, function_learning_model.error.neurons,
                          transform=[ [ -1 ] ] * function_learning_model.error.n_neurons )
        
        # -- probes
        function_learning_model.ground_truth_probe = nengo.Probe( function_learning_model.ground_truth, synapse=0.01 )
        function_learning_model.pre_probe = nengo.Probe( function_learning_model.pre, synapse=0.01 )
        function_learning_model.post_probe = nengo.Probe( function_learning_model.post, synapse=0.01 )
        function_learning_model.error_probe = nengo.Probe( function_learning_model.error, synapse=0.03 )
    
    return function_learning_model


if experiment > 3:
    learned_model_mpes = LearningModel( neurons, dimensions, mPES( gain=gain ), function_to_learn )
    control_model_pes = LearningModel( neurons, dimensions, PES(), function_to_learn )
    control_model_nef = LearningModel( neurons, dimensions, None, function_to_learn )
else:
    learned_model_mpes = LearningConvolutionModel( neurons, dimensions, mPES( gain=gain ), function_to_learn )
    control_model_pes = LearningConvolutionModel( neurons, dimensions, PES(), function_to_learn )
    control_model_nef = LearningConvolutionModel( neurons, dimensions, None, function_to_learn )

# trail runs for each model
errors_iterations_mpes = [ ]
errors_iterations_pes = [ ]
errors_iterations_nef = [ ]
for i in range( iterations ):
    print( "Iteration", i )
    with nengo_dl.Simulator( learned_model_mpes, device=device ) as sim_mpes:
        print( "Learning network (mPES)" )
        sim_mpes.run( sim_time )
    with nengo_dl.Simulator( control_model_pes, device=device ) as sim_pes:
        print( "Control network (PES)" )
        sim_pes.run( sim_time )
    with nengo_dl.Simulator( control_model_nef, device=device ) as sim_nef:
        print( "Control network (NEF)" )
        sim_nef.run( sim_time )
    
    # essential statistics
    num_blocks = int( sim_time / learn_block_time )
    num_testing_blocks = int( num_blocks / 2 )
    for sim, mod, lst in zip( [ sim_mpes, sim_pes, sim_nef ],
                              [ learned_model_mpes, control_model_pes, control_model_nef ],
                              [ errors_iterations_mpes, errors_iterations_pes, errors_iterations_nef ] ):
        # split probe data into the trial run blocks
        ground_truth_data = np.array_split( sim.data[ mod.ground_truth_probe ], sim_time / learn_block_time )
        post_data = np.array_split( sim.data[ mod.post_probe ], sim_time / learn_block_time )
        # extract learning blocks
        learned_ground_truth_data = np.array( [ x for i, x in enumerate( ground_truth_data ) if i % 2 == 0 ] )
        test_ground_truth_data = np.array( [ x for i, x in enumerate( ground_truth_data ) if i % 2 != 0 ] )
        # extract testing blocks
        learned_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 == 0 ] )
        test_post_data = np.array( [ x for i, x in enumerate( post_data ) if i % 2 != 0 ] )
        
        # compute testing error for learn network
        total_error = np.sum( np.sum( np.abs( test_post_data - test_ground_truth_data ), axis=1 ), axis=1 )
        lst.append( total_error )


# 95% confidence interval
def ci( data, confidence=0.95 ):
    from scipy.stats import norm
    
    z = norm.ppf( (1 + confidence) / 2 )
    
    return np.mean( data, axis=0 ), \
           np.mean( data, axis=0 ) + z * np.std( data, axis=0 ) / np.sqrt( len( data ) ), \
           np.mean( data, axis=0 ) - z * np.std( data, axis=0 ) / np.sqrt( len( data ) )


# compute mean testing error and confidence intervals
ci_mpes = ci( errors_iterations_mpes )
ci_pes = ci( errors_iterations_pes )
ci_nef = ci( errors_iterations_nef )

# plot testing error
fig, ax = plt.subplots()
fig.set_size_inches( (12, 8) )
plt.title( exp_name )
x = range( num_testing_blocks )
ax.set_ylabel( "Total error" )
ax.set_xlabel( "Seconds" )
plt.xticks( x, np.array( x ) * sim_time / num_testing_blocks + 2 * learn_block_time )
ax.plot( x, ci_mpes[ 0 ], label="Learned (mPES)", c="g" )
ax.plot( x, ci_mpes[ 1 ], linestyle="--", alpha=0.5, c="g" )
ax.plot( x, ci_mpes[ 2 ], linestyle="--", alpha=0.5, c="g" )
ax.plot( x, ci_pes[ 0 ], label="Control (PES)", c="b" )
ax.plot( x, ci_pes[ 1 ], linestyle="--", alpha=0.5, c="b" )
ax.plot( x, ci_pes[ 2 ], linestyle="--", alpha=0.5, c="b" )
ax.plot( x, ci_nef[ 0 ], label="Control (NEF)", c="r" )
ax.plot( x, ci_nef[ 1 ], linestyle="--", alpha=0.5, c="r" )
ax.plot( x, ci_nef[ 2 ], linestyle="--", alpha=0.5, c="r" )
ax.legend( loc="best" )
fig.show()

# noinspection PyTypeChecker
np.savetxt( dir_data + "results.csv",
            np.squeeze(
                    np.stack(
                            (ci_mpes[ 0 ], ci_mpes[ 1 ], ci_mpes[ 2 ],
                             ci_pes[ 0 ], ci_pes[ 1 ], ci_pes[ 2 ],
                             ci_nef[ 0 ], ci_nef[ 1 ], ci_nef[ 2 ],
                             ),
                            axis=1
                            )
                    ),
            delimiter=",",
            header="Mean mPES error,CI mPES +,CI mPES -,"
                   "Mean PES error,CI PES +,CI PES -,"
                   "Mean NEF error,CI NEF +,CI NEF -,",
            comments="" )
print( exp_string )
print( f"Saved results in {dir_data}" )
fig.savefig( dir_images + "product" + ".pdf" )
print( f"Saved plots in {dir_images}" )
