import nengo
import numpy as np


def plot_network( model ):
    from nengo_extras import graphviz
    
    net = graphviz.net_diagram( model )
    from graphviz import Source
    
    s = Source( net, filename="./net.png", format="png" )
    s.view()


class MemristorArray:
    def __init__( self, in_size, out_size, type, r0=100, r1=2.5 * 10**8, a=-0.128, b=-0.522 ):
        # input/output sizes
        self.input_size = in_size
        self.output_size = out_size
        
        assert type == "single" or type == "pair"
        self.type = type
        
        # create memristor array that implement the weights
        self.memristors = np.empty( (self.output_size, self.input_size), dtype=Memristor )
        for i in range( self.output_size ):
            for j in range( self.input_size ):
                if self.type == "single":
                    self.memristors[ i, j ] = Memristor( self.input_size, self.output_size, "excitatory", r0, r1, a, b )
                if self.type == "pair":
                    self.memristors[ i, j ] = MemristorPair( self.input_size, self.output_size, r0, r1, a, b )
    
    def __call__( self, t, x ):
        input_activities = x[ :self.input_size ]
        error = x[ self.input_size: ]
        
        for i in range( self.output_size ):
            for j in range( self.input_size ):
                # save resistance states for later analysis
                self.memristors[ i, j ].save_state()
                
                # did the input neuron j spike in this timestep?
                spiked = True if input_activities[ j ] else False
                if spiked:
                    # update memristor resistance state
                    self.memristors[ i, j ].pulse( t, error )
        
        # query each memristor for its resistance state
        extract_R = lambda x: x.get_state( t, value="conductance", scaled=True )
        extract_R_V = np.vectorize( extract_R )
        new_weights = extract_R_V( self.memristors )
        
        return np.dot( new_weights, input_activities )
    
    def get_components( self ):
        return self.memristors.flatten()
    
    def plot_state( self, sim, value, err_probe=None ):
        import datetime
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        
        # plot memristor resistance and error
        plt.figure()
        # plt.suptitle( datetime.datetime.now().strftime( '%H:%M:%S %d-%m-%Y' ) )
        fig, ax1 = plt.subplots()
        colour = iter( cm.rainbow( np.linspace( 0, 1, self.memristors.size ) ) )
        for i in range( self.memristors.shape[ 0 ] ):
            for j in range( self.memristors.shape[ 1 ] ):
                c = next( colour )
                self.memristors[ i, j ].plot_state( value, i, j, sim.trange(), ax1, c )
        
        if err_probe:
            ax2 = plt.twinx()
            ax2.plot( sim.trange(), sim.data[ err_probe ], c="r", label="Error" )
        # plt.figlegend()
        plt.title( "Memristor " + value )
        plt.show()


class MemristorPair():
    def __init__( self, in_size, out_size, r0=100, r1=2.5 * 10**8, a=-0.128, b=-0.522 ):
        # input/output sizes
        self.input_size = in_size
        self.output_size = out_size
        
        # instantiate memristor pair
        self.mem_plus = Memristor( self.input_size, self.output_size, "excitatory", r0, r1, a, b )
        self.mem_minus = Memristor( self.input_size, self.output_size, "inhibitory", r0, r1, a, b )
    
    def pulse( self, t, err ):
        self.mem_plus.pulse( t, err )
        self.mem_minus.pulse( t, err )
    
    def get_state( self, t, value, scaled=False ):
        return self.mem_plus.get_state( t, value, scaled ) - self.mem_minus.get_state( t, value, scaled )
    
    def save_state( self ):
        self.mem_plus.save_state()
        self.mem_minus.save_state()
    
    def plot_state( self, value, i, j, range, ax1, c ):
        if value == "resistance":
            tmp_plus = self.mem_plus.history
            tmp_minus = self.mem_minus.history
        if value == "conductance":
            tmp_plus = np.divide( 1, self.mem_plus.history )
            tmp_minus = np.divide( 1, self.mem_minus.history )
        ax1.plot( range, tmp_plus, c="y" )
        ax1.plot( range, tmp_minus, c="b" )


class Memristor:
    def __init__( self, in_size, out_size, type, r0=100, r1=2.5 * 10**8, a=-0.128, b=-0.522 ):
        # input/output sizes
        self.input_size = in_size
        self.output_size = out_size
        
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
        # if self.type == "inhibitory":
        #     # initialise memristor to highest resistance state
        #     self.r_curr = self.r_max
        # if self.type == "excitatory":
        #     # initialise memristor to random low resistance state
        #     import random
        #     self.r_curr = random.randrange( 5.0 * 10**7, 15.0 * 10**7 )
        
        # Weight initialisation
        import random
        self.r_curr = random.randrange( 5.0 * 10**7, 15.0 * 10**7 )
        # self.r_curr = self.r_max
    
    # pulse the memristor with a tension
    def pulse( self, t, err, V=0.01 ):
        # TODO calculate voltage pulse based on magnitude of error?
        # TODO adaptive voltage magnitude?
        
        if (err > 0 and self.type == "excitatory") or (err < 0 and self.type == "inhibitory"):
            c = self.a + self.b * V
            self.r_curr = self.r_min + self.r_max * (((self.r_curr - self.r_min) / self.r_max)**(1 / c) + 1)**c
            # import random
            # self.r_curr = random.uniform( np.finfo( float ).eps, 1.0 )
            
            return self.r_curr
    
    def get_state( self, t, value="conductance", scaled=False, scale_factor=10**4 ):
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
        
        return scale_factor * ret_val
    
    def save_state( self ):
        self.history.append( self.r_curr )
    
    def plot_state( self, value, i, j, range, ax1, c ):
        if value == "resistance":
            tmp = self.history
        if value == "conductance":
            tmp = np.divide( 1, self.history )
        
        ax1.plot( range, tmp, c=c )
