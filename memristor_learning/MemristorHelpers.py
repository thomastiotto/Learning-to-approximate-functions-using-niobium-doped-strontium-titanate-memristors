import datetime
import os
import pickle
import pprint
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import nengo
import numpy as np
from tabulate import tabulate


def make_timestamped_dir( root=None ):
    if root is None:
        root = "../data/"
    
    time_string = datetime.datetime.now().strftime( "%d-%m-%Y_%H-%M" )
    dir_name = root + time_string + "/"
    if os.path.isdir( dir_name ):
        dir_name = dir_name[ :-1 ]
        minutes = str( int( dir_name[ -1 ] ) + 1 )
        dir_name = dir_name[ :-1 ] + minutes + "/"
    dir_images = dir_name + "images/"
    os.mkdir( dir_name )
    os.mkdir( dir_images )
    
    return dir_name, dir_images


def write_experiment_to_file( params, res, headers, table, dir_name, dir_images ):
    with open( dir_name + "info.txt", "w+" ) as f:
        f.write( tabulate( table, headers=headers, tablefmt="github" ) )
        f.write( "\n\nPARAMS:\n" + pprint.pformat( params, width=1 ) )
        f.write( "\n\nSTATS:\n" + pprint.pformat( res[ "stats" ], width=1 ) )
        f.write( f"\n\nMSE: {res[ 'mse' ]} " )
        f.write( f"\nSPARSITY: {res[ 'initial_sparsity' ]} -> {res[ 'end_sparsity' ]} " )
    for key in res:
        if re.match( r'fig', key ):
            if isinstance( res[ key ], list ):
                for i, fig in enumerate( res[ key ] ):
                    fig.savefig( dir_images + key + f"_{i}" )
            else:
                res[ key ].savefig( dir_images + key )
    with open( dir_name + "res.pkl", "wb" ) as f:
        pickle.dump( res, f )


def nested_dict( n, type ):
    if n == 1:
        return defaultdict( type )
    else:
        return defaultdict( lambda: nested_dict( n - 1, type ) )


def generate_encoders( n_neurons, dimensions=1 ):
    if n_neurons % 2 == 0:
        return [ [ -1 ] * int( dimensions ) ] * int( (n_neurons / 2) ) + [ [ 1 ] * int( dimensions ) ] * int(
                (n_neurons / 2) )
    else:
        # toss a coin to fairly decide on the extra central encoder
        centr_enc = -1 if np.random.binomial( 1, 0.5 ) == 0 else 1
        return [ [ -1 ] * int( dimensions ) ] * int( (n_neurons / 2) ) + \
               [ [ centr_enc ] * int( dimensions ) ] \
               + [ [ 1 ] * int( dimensions ) ] * int( (n_neurons / 2) )


def expand_interpolate( oldx, oldy, step=1, include_start=False, include_end=False ):
    from scipy.interpolate import interp1d
    
    start = 1
    end = -1
    if include_start:
        start = 0
    if include_end:
        end = None
    
    try:
        if oldx[ 1 ] - oldx[ 0 ] <= 1:
            return oldx[ 1 ] - oldx[ 0 ], None, None
        
        expanded_interval = np.arange( oldx[ 0 ], oldx[ 1 ] + 1, step=step )
        f = interp1d( oldx, oldy )
        
        return oldx[ 1 ] - oldx[ 0 ], expanded_interval[ start:end ], f( expanded_interval[ start:end ] )
    except IndexError:
        return oldx[ 0 ], None, None


def sparsity_measure( vector ):  # Gini index
    # Max sparsity = 1 (single 1 in the vector)
    if np.all( vector == 0 ):
        return 0
    
    v = np.sort( np.abs( vector ) )
    n = v.shape[ 0 ]
    k = np.arange( n ) + 1
    l1norm = np.sum( v )
    summation = np.sum( (v / l1norm) * ((n - k + 0.5) / n) )
    
    return 1 - 2 * summation


def mse( sim, x, y, learning_time, simulation_step ):
    return (
            np.square( sim.data[ x ][ int( learning_time / simulation_step ): ] -
                       sim.data[ y ][ int( learning_time / simulation_step ): ] )
    ).mean()


def plot_tuning_curves( ens, sim ):
    plt.plot( *nengo.utils.ensemble.tuning_curves( ens, sim ) )
    plt.xlabel( "Input" )
    plt.ylabel( "Firing rate [Hz]" )
    plt.show()


def plot_network( model ):
    from nengo_extras import graphviz
    
    net = graphviz.net_diagram( model )
    from graphviz import Source
    
    s = Source( net, filename="./net.png", format="png" )
    s.view()


def plot_ensemble( sim, ens, time=None ):
    fig = plt.figure()
    for e in ens:
        plt.plot( sim.trange(), sim.data[ e ], c="b", label="Ensemble" )
    if time:
        plt.axvline( x=time, c="k" )
        # ax[ 0 ].annotate( "Training end", xy=(time, np.amax( sim.data[ input ] )), xycoords='data' )
    
    return fig


def plot_pre_post( sim, pre, post, input, error=None, time=None ):
    import matplotlib.pyplot as plt
    
    num_subplots = 1
    if error:
        num_subplots = 2
    fig, axes = plt.subplots( num_subplots, 1, sharex=True, sharey=True, squeeze=False )
    # axes = axes.flatten()
    # plot input, neural representations and error
    # plt.suptitle( datetime.datetime.now().strftime( '%H:%M:%S %d-%m-%Y' ) )
    for i in input:
        axes[ 0, 0 ].plot( sim.trange(), sim.data[ i ], c="k", label="Input" )
    axes[ 0, 0 ].plot( sim.trange(), sim.data[ pre ], c="b", label="Pre" )
    axes[ 0, 0 ].plot( sim.trange(), sim.data[ post ], c="g", label="Post" )
    if error:
        axes[ 1, 0 ].plot( sim.trange(), error, c="r", label="Error" )
    if time:
        for ax in axes:
            ax[ 0 ].axvline( x=time, c="k" )
            # ax[ 0 ].annotate( "Training end", xy=(time, np.amax( sim.data[ input ] )), xycoords='data' )
    plt.legend( loc='best' )
    # plt.show()
    
    return fig


def plot_ensemble_spikes( sim, name, ensemble, input=None, time=None ):
    from nengo.utils.matplotlib import rasterplot
    import matplotlib.pyplot as plt
    
    # plot spikes from pre
    fig = plt.figure()
    # plt.suptitle( datetime.datetime.now().strftime( '%H:%M:%S %d-%m-%Y' ) )
    fig, ax1 = plt.subplots()
    ax1 = plt.subplot( 1, 1, 1 )
    rasterplot( sim.trange(), sim.data[ ensemble ], ax1 )
    ax1.set_xlim( 0, max( sim.trange() ) )
    ax1.set_ylabel( 'Neuron' )
    ax1.set_xlabel( 'Time (s)' )
    if input:
        ax2 = plt.twinx()
        ax2.plot( sim.trange(), sim.data[ input ], c="k" )
    if time:
        time = int( time )
        for t in range( time ):
            plt.axvline( x=t, c="k" )
    plt.title( name + " neural activity" )
    
    return fig
