import datetime
import os

import matplotlib.pyplot as plt
import nengo
import numpy as np
from nengo.processes import Process
from nengo.utils.matplotlib import rasterplot


class Sines( Process ):
    
    def __init__( self, period=4, **kwargs ):
        super().__init__( default_size_in=0, **kwargs )
        
        self.period = period
    
    def make_step( self, shape_in, shape_out, dt, rng, state ):
        # iteratively build phase shifted sines
        s = "lambda t: ("
        phase_shift = (2 * np.pi) / shape_out[ 0 ]
        for i in range( shape_out[ 0 ] ):
            s += f"np.sin( 1 / {self.period} * 2 * np.pi * t + {i * phase_shift}),"
        s += ")"
        signal = eval( s )
        
        def step_sines( t ):
            return signal( t )
        
        return step_sines


class SwitchInputs( Process ):
    def __init__( self, pre_switch, post_switch, switch_time, **kwargs ):
        assert issubclass( pre_switch.__class__, Process ) and issubclass( post_switch.__class__, Process ), \
            f"Expected two nengo Processes, got ({pre_switch.__class__},{post_switch.__class__}) instead"
        
        super().__init__( default_size_in=0, **kwargs )
        
        self.switch_time = switch_time
        self.preswitch_signal = pre_switch
        self.postswitch_signal = post_switch
    
    def make_step( self, shape_in, shape_out, dt, rng, state ):
        preswitch_step = self.preswitch_signal.make_step( shape_in, shape_out, dt, rng, state )
        postswitch_step = self.postswitch_signal.make_step( shape_in, shape_out, dt, rng, state )
        
        def step_switchinputs( t ):
            return preswitch_step( t ) if t < self.switch_time else postswitch_step( t )
        
        return step_switchinputs


class ConditionalProbe:
    def __init__( self, obj, attr, probe_from ):
        if isinstance( obj, nengo.Ensemble ):
            self.size_out = obj.dimensions
        if isinstance( obj, nengo.Node ):
            self.size_out = obj.size_out
        if isinstance( obj, nengo.Connection ):
            self.size_out = obj.size_out
        
        self.attr = attr
        self.time = probe_from
        self.probed_data = [ [ ] for _ in range( self.size_out ) ]
    
    def __call__( self, t, x ):
        if x.shape != (self.size_out,):
            raise RuntimeError(
                    "Expected dimensions=%d; got shape: %s"
                    % (self.size_out, x.shape)
                    )
        if t > 0 and t > self.time:
            for i, k in enumerate( x ):
                self.probed_data[ i ].append( k )
    
    @classmethod
    def setup( cls, obj, attr=None, probe_from=0 ):
        cond_probe = ConditionalProbe( obj, attr, probe_from )
        output = nengo.Node( cond_probe, size_in=cond_probe.size_out )
        nengo.Connection( obj, output, synapse=0.01 )
        
        return cond_probe
    
    def get_conditional_probe( self ):
        return np.array( self.probed_data ).T


class Plotter():
    def __init__( self, trange, rows, cols, dimensions, learning_time, sampling, plot_size=(12, 8), dpi=80, dt=0.001,
                  pre_alpha=0.3 ):
        self.time_vector = trange
        self.plot_sizes = plot_size
        self.dpi = dpi
        self.n_rows = rows
        self.n_cols = cols
        self.n_dims = dimensions
        self.learning_time = learning_time
        self.sampling = sampling
        self.dt = dt
        self.pre_alpha = pre_alpha
    
    def plot_testing( self, pre, post, smooth=False, mse=None ):
        fig, axes = plt.subplots( 1, 1, sharex=True, sharey=True, squeeze=False )
        fig.set_size_inches( self.plot_sizes )
        
        learning_time = int( (self.learning_time / self.dt) / (self.sampling / self.dt) )
        time = self.time_vector[ learning_time:, ... ]
        pre = pre[ learning_time:, ... ]
        post = post[ learning_time:, ... ]
        
        axes[ 0, 0 ].xaxis.set_tick_params( labelsize='xx-large' )
        axes[ 0, 0 ].yaxis.set_tick_params( labelsize='xx-large' )
        
        if smooth:
            from scipy.signal import savgol_filter
            
            pre = np.apply_along_axis( savgol_filter, 0, pre, window_length=51, polyorder=3 )
            post = np.apply_along_axis( savgol_filter, 0, post, window_length=51, polyorder=3 )
        
        axes[ 0, 0 ].plot(
                time,
                pre,
                # linestyle=":",
                alpha=self.pre_alpha,
                label='Pre' )
        axes[ 0, 0 ].set_prop_cycle( None )
        axes[ 0, 0 ].plot(
                time,
                post,
                label='Post' )
        # if self.n_dims <= 3:
        #     axes[ 0, 0 ].legend(
        #             [ f"Pre dim {i}" for i in range( self.n_dims ) ] +
        #             [ f"Post dim {i}" for i in range( self.n_dims ) ],
        #             loc='best' )
        # axes[ 0, 0 ].set_title( "Pre and post decoded on testing phase", fontsize=16 )
        
        if mse is not None:
            axes[ 0, 0 ].text( 0.85, 0.2, f"MSE: {np.round( mse, 5 )}",
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=axes[ 0, 0 ].transAxes )
        
        plt.tight_layout()
        
        return fig
    
    def plot_results( self, input, pre, post, error, smooth=False, mse=None ):
        fig, axes = plt.subplots( 3, 1, sharex=True, sharey=True, squeeze=False )
        fig.set_size_inches( self.plot_sizes )
        
        for ax in axes.flatten():
            ax.xaxis.set_tick_params( labelsize='xx-large' )
            ax.yaxis.set_tick_params( labelsize='xx-large' )
        
        axes[ 0, 0 ].plot(
                self.time_vector,
                input,
                label='Input',
                linewidth=2.0 )
        # if self.n_dims <= 3:
        #     axes[ 0, 0 ].legend(
        #             [ f"Input dim {i}" for i in range( self.n_dims ) ],
        #             loc='best' )
        axes[ 0, 0 ].set_title( "Input signal", fontsize=16 )
        
        if smooth:
            from scipy.signal import savgol_filter
            
            pre = np.apply_along_axis( savgol_filter, 0, pre, window_length=51, polyorder=3 )
            post = np.apply_along_axis( savgol_filter, 0, post, window_length=51, polyorder=3 )
        
        axes[ 1, 0 ].plot(
                self.time_vector,
                pre,
                # linestyle=":",
                alpha=self.pre_alpha,
                label='Pre' )
        axes[ 1, 0 ].set_prop_cycle( None )
        axes[ 1, 0 ].plot(
                self.time_vector,
                post,
                label='Post' )
        # if self.n_dims <= 3:
        #     axes[ 1, 0 ].legend(
        #             [ f"Pre dim {i}" for i in range( self.n_dims ) ] +
        #             [ f"Post dim {i}" for i in range( self.n_dims ) ],
        #             loc='best' )
        axes[ 1, 0 ].set_title( "Pre and post decoded", fontsize=16 )
        
        if smooth:
            from scipy.signal import savgol_filter
            
            error = np.apply_along_axis( savgol_filter, 0, error, window_length=51, polyorder=3 )
        axes[ 2, 0 ].plot(
                self.time_vector,
                error,
                label='Error' )
        if self.n_dims <= 3:
            axes[ 2, 0 ].legend(
                    [ f"Error dim {i}" for i in range( self.n_dims ) ],
                    loc='best' )
        if mse is not None:
            axes[ 2, 0 ].text( 0.85, 0.2, f"MSE: {np.round( mse, 5 )}",
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=axes[ 2, 0 ].transAxes )
        axes[ 2, 0 ].set_title( "Error", fontsize=16 )
        
        for ax in axes:
            ax[ 0 ].axvline( x=self.learning_time, c="k" )
        
        fig.get_axes()[ 0 ].annotate( f"{self.n_rows} neurons, {self.n_dims} dimensions", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=20
                                      )
        plt.tight_layout()
        
        return fig
    
    def plot_ensemble_spikes( self, name, spikes, decoded ):
        fig, ax1 = plt.subplots()
        fig.set_size_inches( self.plot_sizes )
        ax1 = plt.subplot( 1, 1, 1 )
        rasterplot( self.time_vector, spikes, ax1 )
        ax1.axvline( x=self.learning_time, c="k" )
        ax2 = plt.twinx()
        ax2.plot( self.time_vector, decoded, c="k", alpha=0.3 )
        ax1.set_xlim( 0, max( self.time_vector ) )
        ax1.set_ylabel( 'Neuron' )
        ax1.set_xlabel( 'Time (s)' )
        fig.get_axes()[ 0 ].annotate( name + " neural activity", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=20
                                      )
        
        return fig
    
    def plot_values_over_time( self, pos_memr, neg_memr ):
        fig, axes = plt.subplots( self.n_rows, self.n_cols )
        fig.set_size_inches( self.plot_sizes )
        for i in range( axes.shape[ 0 ] ):
            for j in range( axes.shape[ 1 ] ):
                pos_cond = pos_memr[ ..., i, j ]
                neg_cond = neg_memr[ ..., i, j ]
                axes[ i, j ].plot( pos_cond, c="r" )
                axes[ i, j ].plot( neg_cond, c="b" )
                axes[ i, j ].set_title( f"{j}->{i}" )
                axes[ i, j ].set_yticklabels( [ ] )
                axes[ i, j ].set_xticklabels( [ ] )
                plt.subplots_adjust( hspace=0.7 )
        fig.get_axes()[ 0 ].annotate( "Conductances over time", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=20
                                      )
        # plt.tight_layout()
        
        return fig
    
    def plot_weights_over_time( self, pos_memr, neg_memr ):
        fig, axes = plt.subplots( self.n_rows, self.n_cols )
        fig.set_size_inches( self.plot_sizes )
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
                                      fontsize=20
                                      )
        # plt.tight_layout()
        
        return fig
    
    def plot_weight_matrices_over_time( self, weights, n_cols=5, sample_every=0.001 ):
        n_rows = int( self.learning_time / n_cols ) + 1
        fig, axes = plt.subplots( n_rows, n_cols )
        fig.set_size_inches( self.plot_sizes )
        
        for t, ax in enumerate( axes.flatten() ):
            if t <= self.learning_time:
                ax.matshow( weights[ int( (t / self.dt) / (sample_every / self.dt) ), ... ],
                            cmap=plt.cm.Blues )
                ax.set_title( f"{t}" )
                ax.set_yticklabels( [ ] )
                ax.set_xticklabels( [ ] )
                plt.subplots_adjust( hspace=0.7 )
            else:
                ax.set_axis_off()
        fig.get_axes()[ 0 ].annotate( "Weights over time", (0.5, 0.94),
                                      xycoords='figure fraction', ha='center',
                                      fontsize=18
                                      )
        # plt.tight_layout()
        
        return fig


def make_timestamped_dir( root=None ):
    if root is None:
        root = "../data/"
    
    os.makedirs( os.path.dirname( root ), exist_ok=True )
    
    time_string = datetime.datetime.now().strftime( "%d-%m-%Y_%H-%M-%S" )
    dir_name = root + time_string + "/"
    if os.path.isdir( dir_name ):
        raise FileExistsError( "The directory already exists" )
    dir_images = dir_name + "images/"
    dir_data = dir_name + "data/"
    os.mkdir( dir_name )
    os.mkdir( dir_images )
    os.mkdir( dir_data )
    
    return dir_name, dir_images, dir_data


def mse_to_rho_ratio( mse, rho ):
    return [ i for i in np.array( rho ) / mse ]


def correlations( X, Y ):
    import scipy
    
    pearson_correlations = [ ]
    spearman_correlations = [ ]
    kendall_correlations = [ ]
    for x, y in zip( X.T, Y.T ):
        pearson_correlations.append( scipy.stats.pearsonr( x, y )[ 0 ] )
        spearman_correlations.append( scipy.stats.spearmanr( x, y )[ 0 ] )
        kendall_correlations.append( scipy.stats.kendalltau( x, y )[ 0 ] )
    
    return pearson_correlations, spearman_correlations, kendall_correlations


def gini( array ):
    """Calculate the Gini coefficient of exponent numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin( array ) < 0:
        # Values cannot be negative:
        array -= np.amin( array )
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort( array )
    # Index per array element:
    index = np.arange( 1, array.shape[ 0 ] + 1 )
    # Number of array elements:
    n = array.shape[ 0 ]
    # Gini coefficient:
    return ((np.sum( (2 * index - n - 1) * array )) / (n * np.sum( array )))


def save_weights( path, probe ):
    np.save( path + "weights.npy", probe[ -1 ].T )


def save_memristors_to_csv( dir, pos_memr, neg_memr ):
    num_post = pos_memr.shape[ 0 ]
    num_pre = pos_memr.shape[ 1 ]
    
    pos_memr = pos_memr.reshape( (pos_memr.shape[ 0 ], -1) )
    neg_memr = neg_memr.reshape( (neg_memr.shape[ 0 ], -1) )
    
    header = [ ]
    for i in range( num_post ):
        for j in range( num_pre ):
            header.append( f"{j}->{i}" )
    header = ','.join( header )
    
    np.savetxt( dir + "pos_resistances.csv", pos_memr, delimiter=",", header=header, comments="" )
    np.savetxt( dir + "neg_resistances.csv", neg_memr, delimiter=",", header=header, comments="" )
    np.savetxt( dir + "weights.csv", 1 / pos_memr - 1 / neg_memr, delimiter=",", header=header, comments="" )


def save_results_to_csv( dir, input, pre, post, error ):
    header = [ ]
    header.append( ",".join( [ "input" + str( i ) for i in range( input.shape[ 1 ] ) ] ) )
    header.append( ",".join( [ "pre" + str( i ) for i in range( pre.shape[ 1 ] ) ] ) )
    header.append( ",".join( [ "post" + str( i ) for i in range( post.shape[ 1 ] ) ] ) )
    header.append( ",".join( [ "error" + str( i ) for i in range( error.shape[ 1 ] ) ] ) )
    header = ",".join( header )
    
    with open( dir + "results.csv", "w" ) as f:
        np.savetxt( f, np.hstack( (input, pre, post, error) ), delimiter=",", header=header, comments="" )


def nested_dict( n, type ):
    from collections import defaultdict
    
    if n == 1:
        return defaultdict( type )
    else:
        return defaultdict( lambda: nested_dict( n - 1, type ) )
