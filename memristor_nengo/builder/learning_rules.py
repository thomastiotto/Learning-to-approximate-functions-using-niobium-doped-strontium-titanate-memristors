import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble
from nengo.exceptions import BuildError
from memristor_nengo.learning_rules import mPES
from nengo.node import Node


class SimmPES( Operator ):
    def __init__(
            self,
            pre_filtered,
            error,
            delta,
            learning_rate,
            encoders,
            weights,
            states,
            tag=None
            ):
        super( SimmPES, self ).__init__( tag=tag )
        
        self.learning_rate = learning_rate
        
        self.sets = [ ] + states
        self.incs = [ ]
        self.reads = [ pre_filtered, error, encoders, weights ]
        self.updates = [ delta ]
    
    @property
    def pulse_number( self ):
        return self.sets[ 0 ]
    
    @property
    def pre_filtered( self ):
        return self.reads[ 0 ]
    
    @property
    def error( self ):
        return self.reads[ 1 ]
    
    @property
    def encoders( self ):
        return self.reads[ 2 ]
    
    @property
    def weights( self ):
        return self.reads[ 3 ]
    
    @property
    def delta( self ):
        return self.updates[ 0 ]
    
    def _descstr( self ):
        return "pre=%s, error=%s, weights=%s -> %s" % (self.pre_filtered, self.error, self.weights, self.delta)
    
    def make_step( self, signals, dt, rng ):
        pre_filtered = signals[ self.pre_filtered ]
        error = signals[ self.error ]
        delta = signals[ self.delta ]
        n_neurons = pre_filtered.shape[ 0 ]
        alpha = -self.learning_rate * dt / n_neurons
        encoders = signals[ self.encoders ]
        weights = signals[ self.weights ]
        pulse_number = signals[ self.pulse_number ]
        
        def find_spikes( input_activities, shape, output_activities=None ):
            input_size = shape[ 1 ]
            output_size = shape[ 0 ]
            
            spiked_pre = np.tile(
                    np.array( np.rint( input_activities ), dtype=bool ), (output_size, 1)
                    )
            spiked_post = np.tile(
                    np.expand_dims(
                            np.array( np.rint( output_activities ), dtype=bool ), axis=1 ), (1, input_size)
                    ) if output_activities is not None else np.ones( (1, input_size) )
            
            return np.logical_and( spiked_pre, spiked_post )
        
        def step_simmpes():
            # TODO pass everything into conductances
            # TODO pass parameters or equations/functions directly
            # TODO keep track of pulse numbers in exponent Signal
            a = -0.128
            # TODO learning works but reverse bias has basically no effect
            c = -1e-3
            r_min = 1e2
            r_max = 2.5e8
            r_3 = 1e9
            
            # calculate the magnitude of the update based on PES learning rule
            local_error = alpha * np.dot( encoders, error )
            pes_delta = np.outer( local_error, pre_filtered )
            
            # find neurons that spiked and set the other updates to zero
            # spiked_map = find_spikes( pre_filtered, pes_delta.shape )
            # np.logical_and( pes_delta, spiked_map, out=pes_delta )
            
            # analytical derivative of pulse number
            def deriv( n, a ):
                return a * n**(a - 1)
            
            # set update direction and magnitude (unused with powerlaw memristor equations)
            V = np.sign( pes_delta ) * 1e-1
            # use the correct equation for the bidirectional powerlaw memristor update
            # I am using forward difference 1st order approximation to calculate the update delta
            try:
                delta[ V > 0 ] = r_max * deriv( (((weights[ V > 0 ] - r_min) / r_max)**(1 / a)), a )
                delta[ V < 0 ] = -r_3 * deriv( (((r_3 - weights[ V < 0 ]) / r_3)**(1 / c)), c )
            except:
                pass
        
        return step_simmpes


def get_post_ens( conn ):
    """Get the output `.Ensemble` for connection."""
    return (
            conn.post_obj
            if isinstance( conn.post_obj, (Ensemble, Node) )
            else conn.post_obj.ensemble
    )


def build_or_passthrough( model, obj, signal ):
    """Builds the obj on signal, or returns the signal if obj is None."""
    return signal if obj is None else model.build( obj, signal )


@Builder.register( mPES )
def build_mpes( model, mpes, rule ):
    conn = rule.connection
    
    # NB "mpes" is the "mPES()" frontend class
    
    # Create input error signal
    error = Signal( shape=rule.size_in, name="mPES:memristors" )
    model.add_op( Reset( error ) )
    model.sig[ rule ][ "in" ] = error  # error connection will attach here
    
    # Filter pre-synaptic activities with pre_synapse
    acts = build_or_passthrough( model, mpes.pre_synapse, model.sig[ conn.pre_obj ][ "out" ] )
    
    post = get_post_ens( conn )
    encoders = model.sig[ post ][ "encoders" ][ :, conn.post_slice ]
    
    out_size = encoders.shape[ 0 ]
    in_size = acts.shape[ 0 ]
    
    # model.sig[ conn ][ "weights" ] = mpes.initial_normalized_conductances( 1e8, 1.1e8, (out_size, in_size) )
    
    model.sig[ rule ][ "pulse_number" ] = Signal( shape=(out_size, in_size),
                                                  name="%s.pulse_number" % rule,
                                                  # initial_value=
                                                  )
    
    model.add_op(
            SimmPES(
                    acts,
                    error,
                    model.sig[ rule ][ "delta" ],
                    mpes.learning_rate,
                    encoders,
                    model.sig[ conn ][ "weights" ],
                    states=[
                            model.sig[ rule ][ "pulse_number" ]
                            ]
                    )
            )
    
    # expose these for probes
    model.sig[ rule ][ "error" ] = error
    model.sig[ rule ][ "activities" ] = acts
