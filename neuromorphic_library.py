import nengo
import numpy as np
from numpy.core._multiarray_umath import ndarray


def sparsity_measure( vector ):  # Gini index
    # Max sparsity = 1 (single 1 in the vector)
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


def plot_network( model ):
    from nengo_extras import graphviz
    
    net = graphviz.net_diagram( model )
    from graphviz import Source
    
    s = Source( net, filename="./net.png", format="png" )
    s.view()


def plot_pre_post( sim, pre, post, input, error=None, time=None ):
    import datetime
    import matplotlib.pyplot as plt
    
    num_subplots = 1
    if error:
        num_subplots = 2
    fix, axes = plt.subplots( num_subplots, 1, sharex=True, sharey=True, squeeze=False )
    # axes = axes.flatten()
    # plot input, neural representations and error
    # plt.suptitle( datetime.datetime.now().strftime( '%H:%M:%S %d-%m-%Y' ) )
    axes[ 0, 0 ].plot( sim.trange(), sim.data[ input ], c="k", label="Input" )
    axes[ 0, 0 ].plot( sim.trange(), sim.data[ pre ], c="b", label="Pre" )
    axes[ 0, 0 ].plot( sim.trange(), sim.data[ post ], c="g", label="Post" )
    if error:
        axes[ 1, 0 ].plot( sim.trange(), error, c="r", label="Error" )
    if time:
        for ax in axes:
            ax[ 0 ].axvline( x=time, c="k" )
            # ax[ 0 ].annotate( "Training end", xy=(time, np.amax( sim.data[ input ] )), xycoords='data' )
    plt.legend( loc='best' )
    plt.show()


def plot_ensemble_spikes( sim, name, ensemble, input=None, time=None ):
    import datetime
    from nengo.utils.matplotlib import rasterplot
    import matplotlib.pyplot as plt
    
    # plot spikes from pre
    plt.figure()
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
    plt.show()


# TODO add optional paramenters
# TODO or implement learning rules as objects and pass them together with parameters
class MemristorArray:
    def __init__( self, learning_rule, in_size, out_size, dimensions,
                  type="pair", dt=0.001, post_encoders=None, logging=False, beta=1.0 ):
        
        self.input_size = in_size
        self.pre_dimensions = dimensions[ 0 ]
        self.post_dimensions = dimensions[ 1 ]
        self.output_size = out_size
        
        self.dt = dt
        
        if learning_rule == "mOja":
            self.learning_rule = self.mOja
            # self.theta_filter = nengo.Lowpass( tau=1.0 )
            self.learning_rate = 1e-6
            self.beta = beta
        if learning_rule == "mBCM":
            self.learning_rule = self.mBCM
            # self.theta_filter = nengo.Lowpass( tau=1.0 )
            self.learning_rate = 1e-9
        if learning_rule == "mPES":
            assert post_encoders is not None
            self.learning_rule = self.mPES
            self.learning_rate = 10e-4
            self.encoders = post_encoders
        
        # save for analysis
        self.logging = logging
        self.weight_history = [ ]
        self.error_history = [ ]
        
        # to hold future weights
        self.weights = np.zeros( (self.output_size, self.input_size), dtype=np.float )
        
        assert type == "single" or type == "pair"
        # create memristor array that implement the weights
        self.memristors = np.empty( (self.output_size, self.input_size), dtype=Memristor )
        for i in range( self.output_size ):
            for j in range( self.input_size ):
                if type == "single":
                    self.memristors[ i, j ] = Memristor( "excitatory" )
                if type == "pair":
                    self.memristors[ i, j ] = MemristorPair( self.input_size, self.output_size )
                self.weights[ i, j ] = self.memristors[ i, j ].get_state( value="conductance", scaled=True )
    
    def __call__( self, t, x ):
        return self.learning_rule( t, x )
    
    def mOja( self, t, x ):
        input_activities = x[ :self.input_size ]
        output_activities = x[ self.input_size:self.input_size + self.output_size ]
        alpha = self.learning_rate * self.dt
        beta = self.beta
        
        post_squared = alpha * output_activities * output_activities
        forgetting = -beta * self.weights * np.expand_dims( post_squared, axis=1 )
        hebbian = np.outer( alpha * output_activities, input_activities )
        update_direction = hebbian - forgetting
        
        if self.logging:
            # self.history.append( np.sign( update ) )
            self.save_state()
        
        # squash spikes to False (0) or True (100/1000 ...) or everything is always adjusted
        spiked_pre = np.tile(
                np.array( np.rint( input_activities ), dtype=bool ), (self.output_size, 1)
                )
        spiked_post = np.tile(
                np.expand_dims( np.array( np.rint( output_activities ), dtype=bool ), axis=1 ), (1, self.input_size)
                )
        spiked_map = np.logical_and( spiked_pre, spiked_post )
        
        # we only need to update the weights for the neurons that spiked so we filter
        if spiked_map.any():
            for j, i in np.transpose( np.where( spiked_map ) ):
                self.weights[ j, i ] = self.memristors[ j, i ].pulse( update_direction[ j, i ],
                                                                      value="conductance",
                                                                      method="same"
                                                                      )
        
        if self.logging:
            self.weight_history.append( self.weights.copy() )
        
        # calculate the output at this timestep
        return np.dot( self.weights, input_activities )
    
    def mBCM( self, t, x ):
        input_activities = x[ :self.input_size ]
        output_activities = x[ self.input_size:self.input_size + self.output_size ]
        theta = x[ self.input_size + self.output_size: ]
        alpha = self.learning_rate * self.dt
        
        update_direction = output_activities - theta
        # function \phi( a, \theta ) that is the moving threshold
        update = alpha * output_activities * update_direction
        
        if self.logging:
            # self.history.append( np.sign( update ) )
            self.save_state()
        
        # squash spikes to False (0) or True (100/1000 ...) or everything is always adjusted
        spiked_pre = np.tile(
                np.array( np.rint( input_activities ), dtype=bool ), (self.output_size, 1)
                )
        spiked_post = np.tile(
                np.expand_dims( np.array( np.rint( output_activities ), dtype=bool ), axis=1 ), (1, self.input_size)
                )
        spiked_map = np.logical_and( spiked_pre, spiked_post )
        
        # we only need to update the weights for the neurons that spiked so we filter
        if spiked_map.any():
            for j, i in np.transpose( np.where( spiked_map ) ):
                self.weights[ j, i ] = self.memristors[ j, i ].pulse( update_direction[ j ],
                                                                      value="conductance",
                                                                      method="same"
                                                                      )
        
        if self.logging:
            self.weight_history.append( self.weights.copy() )
        
        # calculate the output at this timestep
        return np.dot( self.weights, input_activities )
    
    # TODO can I remove the inverse method from pulse?
    def mPES( self, t, x ):
        input_activities = x[ :self.input_size ]
        # squash error to zero under a certain threshold or learning rule keeps running indefinitely
        error = x[ self.input_size: ] if abs( x[ self.input_size: ] ) > 10**-5 else 0
        alpha = self.learning_rate * self.dt / self.input_size
        
        # we are adjusting weights so calculate local error
        local_error = alpha * np.dot( self.encoders, error )
        
        if self.logging:
            self.error_history.append( error )
            self.save_state()
        
        # squash spikes to False (0) or True (100/1000 ...) or everything is always adjusted
        spiked_map = np.tile(
                np.array( np.rint( input_activities ), dtype=bool ), (self.output_size, 1)
                )
        
        # we only need to update the weights for the neurons that spiked so we filter for their columns
        if spiked_map.any():
            for j, i in np.transpose( np.where( spiked_map ) ):
                self.weights[ j, i ] = self.memristors[ j, i ].pulse( local_error[ j ],
                                                                      value="conductance",
                                                                      method="inverse"
                                                                      )
        
        if self.logging:
            self.weight_history.append( self.weights.copy() )
        
        # calculate the output at this timestep
        return np.dot( self.weights, input_activities )
    
    def get_components( self ):
        return self.memristors.flatten()
    
    def save_state( self ):
        for j in range( self.output_size ):
            for i in range( self.input_size ):
                self.memristors[ j, i ].save_state()
    
    def plot_state( self, sim, value, err_probe=None, combined=False, time=None ):
        import datetime
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        from nengo.utils.matplotlib import rasterplot
        
        # plot memristor resistance and error
        plt.figure()
        # plt.suptitle( datetime.datetime.now().strftime( '%H:%M:%S %d-%m-%Y' ) )
        if not combined:
            fig, axes = plt.subplots()
        if combined:
            fig, axes = plt.subplots( self.output_size, self.input_size )
        plt.xlabel( "Post neurons on rows\nPre neurons on columns" )
        plt.ylabel( "Post neurons on columns" )
        # fig.suptitle( "Memristor " + value, fontsize=16 )
        colour = iter( cm.rainbow( np.linspace( 0, 1, self.memristors.size ) ) )
        for i in range( self.memristors.shape[ 0 ] ):
            for j in range( self.memristors.shape[ 1 ] ):
                c = next( colour )
                if not combined:
                    self.memristors[ i, j ].plot_state( value, i, j, sim.trange(), axes, c, combined )
                    if time:
                        time = int( time )
                        for t in range( time ):
                            axes.axvline( x=t, c="k" )
                if combined:
                    self.memristors[ i, j ].plot_state( value, i, j, sim.trange(), axes[ i, j ], c, combined )
                    if time:
                        time = int( time )
                        for t in range( time ):
                            axes[ i, j ].axvline( x=t, c="k" )
        if err_probe:
            ax2 = plt.twinx()
            ax2.plot( sim.trange(), sim.data[ err_probe ], c="r", label="Error" )
        plt.show()
    
    def plot_weight_matrix( self, time ):
        import matplotlib.pyplot as plt
        
        weights_at_time = self.weight_history[ int( time / self.dt ) ]
        
        fig, ax = plt.subplots()
        
        ax.matshow( weights_at_time, cmap=plt.cm.Blues )
        max_weight = np.amax( weights_at_time )
        min_weight = np.amin( weights_at_time )
        
        for i in range( weights_at_time.shape[ 0 ] ):
            for j in range( weights_at_time.shape[ 1 ] ):
                c = round( (weights_at_time[ j, i ] - min_weight) / (max_weight - min_weight), 2 )
                ax.text( i, j, str( c ), va='center', ha='center' )
        plt.title( "Weights at t=" + str( time ) )
        plt.show()
    
    def get_history( self, select ):
        if select == "weight":
            return self.weight_history
        if select == "error":
            return self.error_history


class MemristorPair():
    def __init__( self, in_size, out_size ):
        # input/output sizes
        self.input_size = in_size
        self.output_size = out_size
        
        # instantiate memristor pair
        self.mem_plus = Memristor( "excitatory" )
        self.mem_minus = Memristor( "inhibitory" )
    
    def pulse( self, adj, value, method, scaled=True ):
        if method == "same":
            if adj > 0:
                self.mem_plus.pulse()
            if adj < 0:
                self.mem_minus.pulse()
        if method == "inverse":
            if adj < 0:
                self.mem_plus.pulse()
            if adj > 0:
                self.mem_minus.pulse()
        
        return self.mem_plus.get_state( value, scaled ) - self.mem_minus.get_state( value, scaled )
    
    def get_state( self, value, scaled ):
        return (self.mem_plus.get_state( value, scaled ) - self.mem_minus.get_state( value, scaled ))
    
    def save_state( self ):
        self.mem_plus.save_state()
        self.mem_minus.save_state()
    
    def plot_state( self, value, i, j, range, ax, c, combined=False ):
        if value == "resistance":
            tmp_plus = self.mem_plus.history
            tmp_minus = self.mem_minus.history
        if value == "conductance":
            tmp_plus = np.divide( 1, self.mem_plus.history )
            tmp_minus = np.divide( 1, self.mem_minus.history )
        ax.plot( range, tmp_plus, c="r", label='Excitatory' )
        ax.plot( range, tmp_minus, c="b", label='Inhibitory' )
        if not combined:
            ax.annotate( str( j + 1 ) + "->" + str( i + 1 ), xy=(range[ 0 ], tmp_plus[ 0 ]), c="r" )
            ax.annotate( str( j + 1 ) + "->" + str( i + 1 ), xy=(range[ 0 ], tmp_minus[ 0 ]), c="b" )
        if combined:
            ax.set_title( str( j + 1 ) + "->" + str( i + 1 ) )
            ax.label_outer()
            ax.set_yticklabels( [ ] )


class Memristor:
    def __init__( self, type, r0=100, r1=2.5 * 10**8, a=-0.128, b=-0.522 ):
        # set parameters of device
        self.r_min = r0
        self.r_max = r1
        self.a = a
        self.b = b
        
        self.n = 0
        # save resistance history for later analysis
        self.history = [ ]
        
        assert type == "inhibitory" or type == "excitatory"
        self.type = type
        if self.type == "inhibitory":
            # initialise memristor to highest resistance state
            self.r_curr = self.r_max
        if self.type == "excitatory":
            # initialise memristor to random low resistance state
            import random
            self.r_curr = random.randrange( 5.0 * 10**7, 15.0 * 10**7 )
        
        # Weight initialisation
        import random
        self.r_curr = random.uniform( 10**8, 2.5 * 10**8 )
        # self.r_curr = self.r_max
    
    # pulse the memristor with a tension
    def pulse( self, V=1e-1 ):
        c = self.a + self.b * V
        self.r_curr = self.r_min + self.r_max * (((self.r_curr - self.r_min) / self.r_max)**(1 / c) + 1)**c
        
        return self.r_curr
    
    def get_state( self, value="conductance", scaled=True, gain=10**4 ):
        epsilon = np.finfo( float ).eps
        
        if value == "conductance":
            g_curr = 1.0 / self.r_curr
            g_min = 1.0 / self.r_max
            g_max = 1.0 / self.r_min
            if scaled:
                ret_val = ((g_curr - g_min) / (g_max - g_min)) + epsilon
            else:
                ret_val = g_curr + epsilon
        
        if value == "resistance":
            if scaled:
                ret_val = ((self.r_curr - self.r_min) / (self.r_max - self.r_min)) + epsilon
            else:
                ret_val = self.r_curr + epsilon
        
        return gain * ret_val
    
    def save_state( self ):
        self.history.append( self.r_curr )
    
    def plot_state( self, value, i, j, range, ax, c ):
        if value == "resistance":
            tmp = self.history
        if value == "conductance":
            tmp = np.divide( 1.0, self.history )
        
        ax.plot( range, tmp, c=c )
        ax.annotate( "(" + str( i ), xy=(10, 10) )
