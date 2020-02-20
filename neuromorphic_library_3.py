import nengo
import numpy as np


def plot_network(model):
    from nengo_extras import graphviz
    
    net = graphviz.net_diagram( model )
    from graphviz import Source
    
    s = Source( net, filename="./net.png", format="png" )
    s.view()
    
    
class MemristorArray:
    def __init__(self, in_size, out_size, type, r0=100, r1=2.5 * 10 ** 8, a=-0.128, b=-0.522 ):
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
                    self.memristors[ i, j ].pulse( error )
        
        # query each memristor for its resistance state
        extract_R = lambda x: x.get_state( value="conductance", scaled=True )
        extract_R_V = np.vectorize( extract_R )
        new_weights = extract_R_V(self.memristors)
        
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
        for i in range(self.memristors.shape[0]):
            for j in range(self.memristors.shape[1]):
                c = next( colour )
                self.memristors[i, j].plot_state( value, i, j, sim.trange(), ax1, c )

        if err_probe:
            ax2 = plt.twinx()
            ax2.plot( sim.trange(), sim.data[ err_probe ], c="r", label="Error" )
        # plt.figlegend()
        plt.title( "Memristor " + value )
        plt.show()


class MemristorPair():
    def __init__(self, in_size, out_size, r0=100, r1=2.5 * 10 ** 8, a=-0.128, b=-0.522):
        # input/output sizes
        self.input_size = in_size
        self.output_size = out_size
        
        # instantiate memristor pair
        self.mem_plus = Memristor( self.input_size, self.output_size, "excitatory", r0, r1, a, b )
        self.mem_minus = Memristor( self.input_size, self.output_size, "inhibitory", r0, r1, a, b )
        
    def pulse( self, err ):
        self.mem_plus.pulse( err )
        self.mem_minus.pulse( err )
    
    def get_state( self, value, scaled=False ):
        return self.mem_plus.get_state( value, scaled ) - self.mem_minus.get_state( value, scaled )
    
    def save_state( self ):
        self.mem_plus.save_state()
        self.mem_minus.save_state()
    
    def plot_state( self, value, i, j, range, ax1, c ):
        if value == "resistance":
            tmp_plus = self.mem_plus.history
            tmp_minus = self.mem_minus.history
        if value == "conductance":
            tmp_plus = np.divide(1, self.mem_plus.history )
            tmp_minus = np.divide(1, self.mem_minus.history )
        ax1.plot( range, tmp_plus, c="y" )
        ax1.plot( range, tmp_minus, c="b" )
        

class Memristor:
    def __init__( self, in_size, out_size, type, r0=100, r1=2.5 * 10 ** 8, a=-0.128, b=-0.522 ):
        # input/output sizes
        self.input_size = in_size
        self.output_size = out_size

        # set parameters of device
        self.r0 = r0
        self.r1 = r1
        self.a = a
        self.b = b
        
        self.n = 0
        
        # save resistance history for later analysis
        self.history = []
        
        assert type == "inhibitory" or type == "excitatory"
        self.type = type
        # if self.type == "inhibitory":
        #     # initialise memristor to random resistance state
        #     self.R = self.r1
        # else:
        #     # initialise memristor to random low resistance state
        #     import random
        #     self.R = random.randrange( 5.0 * 10 ** 7, 15.0 * 10 ** 7 )
        
        # Weight initialisation
        import random
        self.R = random.randrange( self.r0, self.r1 ) / np.sqrt( self.input_size )
    
        
    # pulse the memristor with a tension
    def pulse( self, err, V=1, k=0.1 ):
        # TODO calculate voltage pulse based on magnitude of error?
        # TODO adaptive learning rate?
        # TODO add memristor scaling?
        m = 100000
        
        if (err > 0 and self.type == "excitatory") or (err < 0 and self.type == "inhibitory"):
            # increment pulse counter
            self.n += 1
            c = self.a + self.b * V
            
            # calculate new resistance state
            # self.R = self.r0 + self.r1 * self.n ** c)
            # update rule
            self.R += k * c * self.r1 * self.n ** (c-1)
        
        return self.R
    
    def get_state( self, value="conductance", scaled=False ):
        epsilon = np.finfo(float).eps
        if value == "conductance":
            G = 1.0 / self.R
            g0 = 1.0 / self.r1
            g1 = 1.0 / self.r0
            if scaled:
                return ( G - g0 ) / ( g1 - g0 ) + epsilon
            else:
                return G + epsilon
        
        if value == "resistance":
            if scaled:
                return ( self.R - self.r0 ) / ( self.r1 - self.r0 ) + epsilon
            else:
                return self.R + epsilon
        
    def save_state( self ):
        self.history.append( self.R )
        
    def plot_state( self, value, i, j, range, ax1, c ):
        if value == "resistance":
            tmp = self.history
        if value == "conductance":
            tmp = np.divide(1, self.history)
        
        ax1.plot( range, tmp, c=c )