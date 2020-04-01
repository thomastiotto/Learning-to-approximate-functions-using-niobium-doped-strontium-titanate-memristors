import numpy as np


class MemristorPair:
    def __init__( self ):
        self.mem_plus = None
        self.mem_minus = None
    
    def pulse( self, V ):
        # TODO grade voltage based on signal coming from LearningRule, for now fixed at 1e-1
        if V > 0:
            self.mem_plus.pulse()
        if V < 0:
            self.mem_minus.pulse()
        
        return self.mem_plus.get_state() - self.mem_minus.get_state()
    
    def get_state( self, value="conductance", scaled=True ):
        return self.mem_plus.get_state( value, scaled ) - self.mem_minus.get_state( value, scaled )
    
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
            # ax.label_outer()
            # ax.set_yticklabels( [ ] )


class MemristorAnoukPair( MemristorPair ):
    def __init__( self ):
        super().__init__()
        # instantiate memristor pair
        self.mem_plus = MemristorAnouk()
        self.mem_minus = MemristorAnouk()


class Memristor:
    def __init__( self ):
        # save resistance history for later analysis
        self.history = [ ]
        
        self.r_curr = None
        self.r_max = None
        self.r_min = None
    
    # pulse the memristor with a tension
    def pulse( self, V=1e-1 ):
        
        pulse_number = self.compute_pulse_number( self.r_curr, V )
        self.r_curr = self.compute_resistance( pulse_number + 1, V )
        
        return self.get_state()
    
    def compute_pulse_number( self, R, V ):
        raise NotImplementedError
    
    def compute_resistance( self, n, V ):
        raise NotImplementedError
    
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
    
    def plot_state( self, value, i, j, range, ax, c, combined=False ):
        if value == "resistance":
            tmp = self.history
        if value == "conductance":
            tmp = np.divide( 1.0, self.history )
        
        ax.plot( range, tmp, c=c )
        ax.annotate( "(" + str( i ), xy=(10, 10) )
    
    def plot_memristor_curve_exhaustive( self, V=1, threshold=1000, step=10, interpolate_after_threshold=True ):
        import matplotlib.pyplot as plt
        import sys, time
        from memristor_learning.MemristorHelpers import expand_interpolate
        
        x = [ ]
        y = [ ]
        n = 1
        it = 1
        r_curr = self.r_max if V >= 0 else self.r_min
        
        def thresh():
            if V >= 0 and r_curr >= self.r_min + threshold:
                return True
            elif V < 0 and r_curr <= self.r_max - threshold:
                return True
            else:
                return False
        
        start_time = time.time()
        while thresh():
            x.append( n )
            y.append( r_curr )
            
            r_curr = self.compute_resistance( n, V )
            sys.stdout.write( f"\rIteration {it}"
                              f", time {round( time.time() - start_time, 2 )}:"
                              f" resistance {self.r_min}/{round( r_curr, 2 )}/{self.r_max}"
                              f", threshold {threshold}"
                              f", step {step}" )
            n += step
            it += 1
            sys.stdout.flush()
        print( f"\n{round( time.time() - start_time, 2 )} seconds elapsed for exhaustive step" )
        
        if interpolate_after_threshold:
            start_time = time.time()
            x_limit = self.r_min + threshold if V >= 0 else self.r_max - threshold
            
            # subdivide remaining range into blocks of equal step size
            oldx = np.array( [ x[ -1 ], x[ -1 ] + round( x_limit / step, 1 ) ] )
            # interpolate from last real value to the theoretical maximum
            oldy = np.array( [ y[ -1 ], self.r_max ] )
            int_length, newx, newy = expand_interpolate( oldx, oldy, step=step, include_end=True )
            
            x = np.array( x )
            x = np.append( x, newx )
            y = np.array( y )
            y = np.append( y, newy )
            
            print( f"\n{round( time.time() - start_time, 2 )} seconds elapsed for interpolation step" )
        
        plt.plot( x, y )
        plt.yscale( 'log' )
        plt.xlabel( "Pulses (n)" )
        plt.ylabel( "Resistance (R)" )
        plt.grid( alpha=.4, linestyle='--' )
        plt.show()
        
        import pickle
        with open( "../data/memristor_curve.pkl", "wb" ) as f:
            pickle.dump( zip( x, y ), f )
    
    def plot_memristor_curve_interpolate( self, V=1, threshold=1000 ):
        import matplotlib.pyplot as plt
        from math import fabs, floor
        from memristor_learning.MemristorHelpers import expand_interpolate
        
        c = self.a + self.b * V
        
        x = [ ]
        y = [ ]
        n = 0
        step = 1
        r_curr = self.r_max
        
        from tqdm import tqdm
        with tqdm( total=self.r_max - (self.r_min + threshold) ) as pbar:
            r_pre = r_curr
            n += step
            r_curr = self.compute_resistance( n, V )
            r_diff = fabs( r_curr - r_pre )
            step += floor( 1 / r_diff )
            
            x.append( n )
            y.append( r_curr )
            
            pbar.update( n )
        
        c = 0
        
        with tqdm( total=x[ -1 ] ) as pbar:
            while c < len( x ):
                int_length, newx, newy = expand_interpolate( x[ c:c + 2 ], y[ c:c + 2 ] )
                if newx is not None:
                    for cc, (xx, yy) in enumerate( zip( newx, newy ) ):
                        x.insert( c + cc + 1, xx )
                        y.insert( c + cc + 1, yy )
                c += int_length
                pbar.update( int_length )
        
        plt.plot( x, y )
        plt.yscale( 'log' )
        plt.xlabel( "Pulses (n)" )
        plt.ylabel( "Resistance (R)" )
        plt.grid( alpha=.4, linestyle='--' )
        plt.show()


class MemristorAnouk( Memristor ):
    def __init__( self, r0=100, r1=2.5 * 10**8, a=-0.128, b=-0.522 ):
        super().__init__()
        # set parameters of device
        self.r_min = r0
        self.r_max = r1
        self.a = a
        self.b = b
        
        # Weight initialisation
        import random
        self.r_curr = random.uniform( 10**8, 2.5 * 10**8 )
        # self.r_curr = self.r_max
    
    def compute_pulse_number( self, R, V ):
        return int( round( ((R - self.r_min) / self.r_max)**(1 / (self.a + self.b * V)), 0 ) )
    
    def compute_resistance( self, n, V ):
        return self.r_min + self.r_max * n**(self.a + self.b * V)


class MemristorAnoukBidirectional( Memristor ):
    def __init__( self, r0=100, r1=2.5 * 10**8, a=-0.128, b=-0.522 ):
        super().__init__()
        # set parameters of device
        self.r_min = r0
        self.r_max = r1
        self.a = a
        self.b = b
        
        # Weight initialisation
        import random
        self.r_curr = random.uniform( 10**8, 2.5 * 10**8 )
        # self.r_curr = self.r_max
    
    def compute_pulse_number( self, R, V ):
        if V >= 0:
            return int( round( ((R - self.r_min) / self.r_max)**(1 / (self.a + self.b * V)), 0 ) )
        else:
            return int( round( ((R - self.r_max) / (self.r_min - self.r_max))**(1 / (self.a + self.b * (-V / 4))), 0 ) )
    
    def compute_resistance( self, n, V ):
        if V >= 0:
            return self.r_min + self.r_max * n**(self.a + self.b * V)
        else:
            return self.r_max - (self.r_max - self.r_min) * n**(self.a + self.b * (-V / 4))
