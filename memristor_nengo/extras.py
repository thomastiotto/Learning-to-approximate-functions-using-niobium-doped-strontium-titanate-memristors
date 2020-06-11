import numpy as np
from nengo.utils.matplotlib import rasterplot
import matplotlib.pyplot as plt

plot_sizes = None
n_rows = None
n_cols = None
time = None


def set_plot_params( trange, rows, cols, plot_size=(12, 8) ):
    global plot_sizes, n_rows, n_cols, time
    
    time = trange
    plot_sizes = plot_size
    n_rows = rows
    n_cols = cols


def plot_results( input, learn, pre, post, error ):
    # Plot the input signal
    plt.figure( figsize=plot_sizes )
    plt.subplot( 3, 1, 1 )
    plt.plot(
            time,
            input,
            label='Input',
            linewidth=2.0 )
    plt.plot(
            time,
            learn,
            label='Stop learning',
            color='r',
            linewidth=2.0 )
    plt.legend( loc='lower left' )
    plt.ylim( -1.2, 1.2 )
    plt.subplot( 3, 1, 2 )
    plt.plot(
            time,
            pre,
            label='Pre' )
    plt.plot(
            time,
            post,
            label='Post' )
    plt.legend( loc='lower left' )
    plt.ylim( -1.2, 1.2 )
    plt.subplot( 3, 1, 3 )
    plt.plot(
            time,
            error,
            label='Error' )
    plt.legend( loc='lower left' )
    plt.tight_layout()


def generate_sines( dimensions ):
    # iteratively build phase shifted sines
    s = "lambda t: ("
    phase_shift = (2 * np.pi) / dimensions
    for i in range( dimensions ):
        s += f"np.sin( 1 / 4 * 2 * np.pi * t + {i * phase_shift}),"
    s += ")"
    
    return eval( s )


def plot_ensemble_spikes( name, spikes, decoded ):
    # plot spikes from pre
    plt.figure()
    # plt.suptitle( datetime.datetime.now().strftime( '%H:%M:%S %d-%m-%Y' ) )
    fig, ax1 = plt.subplots()
    ax1 = plt.subplot( 1, 1, 1 )
    rasterplot( time, spikes, ax1 )
    ax2 = plt.twinx()
    ax2.plot( time, decoded, c="k", alpha=0.3 )
    ax1.set_xlim( 0, max( time ) )
    ax1.set_ylabel( 'Neuron' )
    ax1.set_xlabel( 'Time (s)' )
    plt.title( name + " neural activity" )
    
    return fig


def plot_conductances_over_time( pos_memr, neg_memr ):
    plt.figure( figsize=plot_sizes )
    fig, axes = plt.subplots( n_rows, n_cols )
    for i in range( axes.shape[ 0 ] ):
        for j in range( axes.shape[ 1 ] ):
            pos_cond = 1 / pos_memr[ ..., i, j ]
            neg_cond = 1 / neg_memr[ ..., i, j ]
            axes[ i, j ].plot( pos_cond, c="r" )
            axes[ i, j ].plot( neg_cond, c="b" )
            axes[ i, j ].set_title( f"{j}->{i}" )
    plt.suptitle( "Conductances over time" )
    plt.tight_layout()
    
    return fig


def plot_weights_over_time( pos_memr, neg_memr ):
    plt.figure( figsize=plot_sizes )
    fig, axes = plt.subplots( n_rows, n_cols )
    for i in range( axes.shape[ 0 ] ):
        for j in range( axes.shape[ 1 ] ):
            pos_cond = 1 / pos_memr[ ..., i, j ]
            neg_cond = 1 / neg_memr[ ..., i, j ]
            axes[ i, j ].plot( pos_cond - neg_cond, c="g" )
            axes[ i, j ].set_title( f"{j}->{i}" )
    plt.suptitle( "Weights over time" )
    plt.tight_layout()
    
    return fig


def plot_weight_matrices_over_time( n_cols, learn_time, weights, dt ):
    n_rows = int( learn_time / n_cols )
    plt.figure( figsize=plot_sizes )
    fig, axes = plt.subplots( n_rows, n_cols )
    t = 0
    for i in range( axes.shape[ 0 ] ):
        for j in range( axes.shape[ 1 ] ):
            axes[ i, j ].matshow( weights[ int( t / dt ), ... ], cmap=plt.cm.Blues )
            axes[ i, j ].set_title( f"{t}" )
            print( t )
            print( weights[ int( t / dt ), ... ] )
            t += 1
    plt.suptitle( "Weights over time" )
    plt.tight_layout()
    plt.show()
