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


def plot_results( input, learn, pre, post, error, smooth=False ):
    n_neurons = n_rows
    n_dimensions = pre.shape[ 1 ]
    
    plt.figure( figsize=plot_sizes )
    fig, axes = plt.subplots( 3, 1, sharex=True, sharey=True, squeeze=False )
    axes[ 0, 0 ].plot(
            time,
            input,
            label='Input',
            linewidth=2.0 )
    axes[ 0, 0 ].plot(
            time,
            learn,
            label='Stop learning',
            color='r',
            linewidth=2.0 )
    plt.legend( loc='lower left' )
    
    if smooth:
        from scipy.signal import savgol_filter
        
        pre = np.apply_along_axis( savgol_filter, 0, pre, window_length=51, polyorder=3 )
        post = np.apply_along_axis( savgol_filter, 0, post, window_length=51, polyorder=3 )
    
    axes[ 1, 0 ].plot(
            time,
            pre,
            linestyle=":",
            label='Pre' )
    axes[ 1, 0 ].set_prop_cycle( None )
    axes[ 1, 0 ].plot(
            time,
            post,
            label='Post' )
    if n_dimensions <= 3:
        axes[ 1, 0 ].legend(
                [ f"Pre dim {i}" for i in range( n_dimensions ) ] +
                [ f"Post dim {i}" for i in range( n_dimensions ) ],
                loc='best' )
    
    if smooth:
        from scipy.signal import savgol_filter
        
        error = np.apply_along_axis( savgol_filter, 0, error, window_length=51, polyorder=3 )
    axes[ 2, 0 ].plot(
            time,
            error,
            label='Error' )
    if n_dimensions <= 3:
        axes[ 2, 0 ].legend(
                [ f"Error dim {i}" for i in range( n_dimensions ) ],
                loc='best' )
    plt.tight_layout()
    plt.suptitle( f"{n_neurons} neurons, {n_dimensions} dimensions" )


def generate_sines( dimensions ):
    # iteratively build phase shifted sines
    s = "lambda t: ("
    phase_shift = (2 * np.pi) / dimensions
    for i in range( dimensions ):
        s += f"np.sin( 1 / 4 * 2 * np.pi * t + {i * phase_shift}),"
    s += ")"
    
    return eval( s )


def plot_ensemble_spikes( name, spikes, decoded ):
    plt.figure( figsize=plot_sizes )
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


def plot_values_over_time( pos_memr, neg_memr ):
    plt.figure( figsize=plot_sizes )
    fig, axes = plt.subplots( n_rows, n_cols )
    for i in range( axes.shape[ 0 ] ):
        for j in range( axes.shape[ 1 ] ):
            pos_cond = pos_memr[ ..., i, j ]
            neg_cond = neg_memr[ ..., i, j ]
            axes[ i, j ].plot( pos_cond, c="r" )
            axes[ i, j ].plot( neg_cond, c="b" )
            axes[ i, j ].set_yticklabels( [ ] )
            axes[ i, j ].set_xticklabels( [ ] )
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
            axes[ i, j ].set_yticklabels( [ ] )
            axes[ i, j ].set_xticklabels( [ ] )
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
            axes[ i, j ].set_yticklabels( [ ] )
            axes[ i, j ].set_xticklabels( [ ] )
            print( t )
            print( weights[ int( t / dt ), ... ] )
            t += 1
    plt.suptitle( "Weights over time" )
    plt.tight_layout()
    
    return fig
