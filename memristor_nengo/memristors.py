from nengo.base import NengoObject
from nengo.params import Default, IntParam, NumberParam, Parameter, FrozenObject
from nengo.dists import DistOrArrayParam
from nengo.dists import Uniform


class MemristorType( FrozenObject ):
    """Base class for Nengo memristor models.

        Attributes
        ----------
        probeable : tuple
            Signals that can be probed in the neuron population.
        Parameters
        ----------
        resistance : int
            The number of neurons.
        dimensions : int
            The number of representational dimensions.
    """
    probeable = ()
    
    r_min = NumberParam( "r_min", low=0, high=2.5e8, default=1e2, readonly=True )
    r_max = NumberParam( "r_max", low=0, high=2.5e8, default=2.5e8, readonly=True )
    
    # r_curr = DistOrArrayParam( "resistance", default=Uniform( 1e8, 1.1e8, integer=True ) )
    
    def __init__( self,
                  r_min=Default,
                  r_max=Default ):
        super().__init__()
        # self.r_min = r_min
        # self.r_max = r_max
    
    def compute_pulse_number( self, R, V ):
        raise NotImplementedError( "Memristors must provide compute_pulse_number" )
    
    def compute_resistance( self, n, V ):
        raise NotImplementedError( "Memristors must provide compute_resistance" )
    
    def step_math( self, output ):
        """Compute the new memristor resistance state given current state.

            Parameters
            ----------
            TBD

            Returns
            -------
            current : (TBD,TBD)
                The new resistance state of the memristor.
        """
        V = 1e-1
        
        pulse_number = self.compute_pulse_number( self.r_curr, V )
        old_r = self.compute_resistance( pulse_number, V )
        self.r_curr = self.compute_resistance( pulse_number + 1, V )
        output[ ... ] = self.r_curr - old_r


class BidirectionalPowerlaw( MemristorType ):
    """Generic bidirectional memristor
    
    Parameters
    ----------
    TBD
    """
    a = NumberParam( "a", default=1e-3, readonly=True )
    c = NumberParam( "c", default=1e-3, readonly=True )
    
    def __init__( self,
                  r_min=Default,
                  r_max=Default,
                  a=Default,
                  c=Default
                  ):
        super().__init__( r_min, r_max )
        
        # self.a = a
        # self.c = c
        self.r_3 = 1e9
    
    def compute_pulse_number( self, R, V ):
        if V >= 0:
            return ((R - self.r_min) / self.r_max)**(1 / self.a)
        else:
            return ((self.r_3 - R) / self.r_3)**(1 / self.c)
    
    def compute_resistance( self, n, V ):
        if V > 0:
            return self.r_min + self.r_max * n**self.a
        elif V < 0:
            return self.r_3 - self.r_3 * n**self.c
        else:
            return self.r_curr


class MemristorTypeParam( Parameter ):
    def coerce( self, instance, memristors ):
        self.check_type( instance, memristors, MemristorType )
        return super().coerce( instance, memristors )
