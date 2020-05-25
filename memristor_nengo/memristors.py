from nengo.base import NengoObject
from nengo.params import BoolParam, Default, IntParam, NumberParam
from nengo.dists import Uniform


class MemristorType( NengoObject ):
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
    probeable = ("resistance")
    
    resistance = IntParam( "resistance", low=0, high=2.5e8, default=Uniform( 1e8, 1.1e8, integer=True ) )
    
    def compute_pulse_number( self, R, V ):
        raise NotImplementedError( "Memristors must provide compute_pulse_number" )
    
    def compute_resistance( self, n, V ):
        raise NotImplementedError( "Memristors must provide compute_resistance" )
    
    def pulse( self, r_curr, output ):
        """Compute the new memristor resistance state given current state.

            Parameters
            ----------

            Returns
            -------
            current : (n_memristors,)
                The new resistance state of the memristor.
        """
        V = 1e-1
        
        pulse_number = self.compute_pulse_number( r_curr, V )
        output[ ... ] = self.compute_resistance( pulse_number + 1, V )
