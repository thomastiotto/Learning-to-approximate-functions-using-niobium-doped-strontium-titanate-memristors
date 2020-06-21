import numpy as np
from nengo.utils.matplotlib import rasterplot
import matplotlib.pyplot as plt


class Plotter():
    def __init__( self, trange, rows, cols, dimensions, learning_time, plot_size=(12, 8), dt=0.001 ):
        self.time_vector = trange
        self.plot_sizes = plot_size
        self.n_rows = rows
        self.n_cols = cols
        self.n_dims = dimensions
        self.learning_time = learning_time
        self.dt = dt
    
    def plot_results( self, input, pre, post, error, smooth=False, mse=None ):
        plt.figure( figsize=self.plot_sizes )
        fig, axes = plt.subplots( 3, 1, sharex=True, sharey=True, squeeze=False )
        axes[ 0, 0 ].plot(
                self.time_vector,
                input,
                label='Input',
                linewidth=2.0 )
        if self.n_dims <= 3:
            axes[ 0, 0 ].legend(
                    [ f"Input dim {i}" for i in range( self.n_dims ) ],
                    loc='best' )
        axes[ 0, 0 ].set_title( "Input signal" )
        
        if smooth:
            from scipy.signal import savgol_filter
            
            pre = np.apply_along_axis( savgol_filter, 0, pre, window_length=51, polyorder=3 )
            post = np.apply_along_axis( savgol_filter, 0, post, window_length=51, polyorder=3 )
        
        axes[ 1, 0 ].plot(
                self.time_vector,
                pre,
                linestyle=":",
                label='Pre' )
        axes[ 1, 0 ].set_prop_cycle( None )
        axes[ 1, 0 ].plot(
                self.time_vector,
                post,
                label='Post' )
        if self.n_dims <= 3:
            axes[ 1, 0 ].legend(
                    [ f"Pre dim {i}" for i in range( self.n_dims ) ] +
                    [ f"Post dim {i}" for i in range( self.n_dims ) ],
                    loc='best' )
        axes[ 1, 0 ].set_title( "Pre and post decoded" )
        
        if smooth:
            from scipy.signal import savgol_filter
            
            error = np.apply_along_axis( savgol_filter, 0, error, window_length=51, polyorder=3 )
        axes[ 2, 0 ].plot(
                self.time_vector,
                error,
                label='Error' )
        # trendline
        # z = np.polyfit( time_vector, error, 1 )
        # p = np.poly1d( np.squeeze( z ) )
        # axes[ 2, 0 ].plot(
        #         time_vector,
        #         p( time_vector ),
        #         label='Error trend',
        #         c="k",
        #         linestyle=":" )
        if self.n_dims <= 3:
            axes[ 2, 0 ].legend(
                    [ f"Error dim {i}" for i in range( self.n_dims ) ],
                    loc='best' )
        if mse is not None:
            axes[ 2, 0 ].text( 0.85, 0.2, f"MSE: {np.round( mse, 5 )}",
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=axes[ 2, 0 ].transAxes )
        axes[ 2, 0 ].set_title( "Error" )
        
        for ax in axes:
            ax[ 0 ].axvline( x=self.learning_time, c="k" )
        
        fig.get_axes()[ 0 ].annotate( f"{self.n_rows} neurons, {self.n_dims} dimensions", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=18
                                      )
        plt.tight_layout()
        
        return fig
    
    def plot_ensemble_spikes( self, name, spikes, decoded ):
        plt.figure( figsize=self.plot_sizes )
        fig, ax1 = plt.subplots()
        ax1 = plt.subplot( 1, 1, 1 )
        rasterplot( self.time_vector, spikes, ax1 )
        ax2 = plt.twinx()
        ax2.plot( self.time_vector, decoded, c="k", alpha=0.3 )
        ax1.set_xlim( 0, max( self.time_vector ) )
        ax1.set_ylabel( 'Neuron' )
        ax1.set_xlabel( 'Time (s)' )
        fig.get_axes()[ 0 ].annotate( name + " neural activity", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=18
                                      )
        
        return fig
    
    def plot_values_over_time( self, pos_memr, neg_memr ):
        plt.figure( figsize=self.plot_sizes )
        fig, axes = plt.subplots( self.n_rows, self.n_cols )
        for i in range( axes.shape[ 0 ] ):
            for j in range( axes.shape[ 1 ] ):
                pos_cond = pos_memr[ ..., i, j ]
                neg_cond = neg_memr[ ..., i, j ]
                axes[ i, j ].plot( pos_cond, c="r" )
                axes[ i, j ].plot( neg_cond, c="b" )
                axes[ i, j ].set_yticklabels( [ ] )
                axes[ i, j ].set_xticklabels( [ ] )
                axes[ i, j ].set_title( f"{j}->{i}" )
                plt.subplots_adjust( hspace=0.7 )
        fig.get_axes()[ 0 ].annotate( "Conductances over time", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=18
                                      )
        plt.tight_layout()
        
        return fig
    
    def plot_weights_over_time( self, pos_memr, neg_memr ):
        plt.figure( figsize=self.plot_sizes )
        fig, axes = plt.subplots( self.n_rows, self.n_cols )
        for i in range( axes.shape[ 0 ] ):
            for j in range( axes.shape[ 1 ] ):
                pos_cond = 1 / pos_memr[ ..., i, j ]
                neg_cond = 1 / neg_memr[ ..., i, j ]
                axes[ i, j ].plot( pos_cond - neg_cond, c="g" )
                axes[ i, j ].set_title( f"{j}->{i}" )
                axes[ i, j ].set_yticklabels( [ ] )
                axes[ i, j ].set_xticklabels( [ ] )
                plt.subplots_adjust( hspace=0.7 )
        fig.get_axes()[ 0 ].annotate( "Weights over time", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=18
                                      )
        plt.tight_layout()
        
        return fig
    
    def plot_weight_matrices_over_time( self, weights, n_cols=5, sample_every=0.001 ):
        n_rows = int( self.learning_time / n_cols )
        plt.figure( figsize=self.plot_sizes )
        fig, axes = plt.subplots( n_rows, n_cols )
        t = 0
        for i in range( axes.shape[ 0 ] ):
            for j in range( axes.shape[ 1 ] ):
                axes[ i, j ].matshow( weights[ int( (t / self.dt) / (sample_every / self.dt) ), ... ],
                                      cmap=plt.cm.Blues )
                axes[ i, j ].set_title( f"{t}" )
                axes[ i, j ].set_yticklabels( [ ] )
                axes[ i, j ].set_xticklabels( [ ] )
                plt.subplots_adjust( hspace=0.7 )
                
                t += 1
        fig.get_axes()[ 0 ].annotate( "Weights over time", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=18
                                      )
        plt.tight_layout()
        
        return fig


def generate_sines( dimensions ):
    # iteratively build phase shifted sines
    s = "lambda t: ("
    phase_shift = (2 * np.pi) / dimensions
    for i in range( dimensions ):
        s += f"np.sin( 1 / 4 * 2 * np.pi * t + {i * phase_shift}),"
    s += ")"
    
    return eval( s )
