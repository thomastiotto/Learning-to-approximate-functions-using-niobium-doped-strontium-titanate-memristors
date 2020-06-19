import warnings

from nengo.params import Default, NumberParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.learning_rules import LearningRuleType
from nengo.learning_rules import PES


class mPES( LearningRuleType ):
    modifies = "weights"
    probeable = ("error", "activities", "delta", "pos_memristors", "neg_memristors")
    
    learning_rate = NumberParam( "learning_rate", low=0, readonly=True, default=1e-4 )
    pre_synapse = SynapseParam( "pre_synapse", default=Lowpass( tau=0.005 ), readonly=True )
    r_max = NumberParam( "r_max", readonly=True, default=2.5e8 )
    r_min = NumberParam( "r_min", readonly=True, default=1e2 )
    
    def __init__( self, initial_conductances_pos, initial_conductances_neg, learning_rate=Default, pre_synapse=Default,
                  r_max=Default, r_min=Default, seed=None ):
        super().__init__( learning_rate, size_in="post_state" )
        if learning_rate is not Default and learning_rate >= 1.0:
            warnings.warn(
                    "This learning rate is very high, and can result "
                    "in floating point errors from too much current."
                    )
        
        self.initial_conductances_pos = initial_conductances_pos
        self.initial_conductances_neg = initial_conductances_neg
        
        self.pre_synapse = pre_synapse
        self.r_max = r_max
        self.r_min = r_min
        
        np.random.seed( seed )
    
    def normalized_conductance( self, R ):
        epsilon = np.finfo( float ).eps
        gain = 1e5
        
        g_curr = 1.0 / R
        g_min = 1.0 / self.r_max
        g_max = 1.0 / self.r_min
        
        return gain * (((g_curr - g_min) / (g_max - g_min)) + epsilon)
    
    def initial_normalized_conductances( self, low, high, shape ):
        gain = 1e5
        
        g_curr = 1.0 / np.random.uniform( low, high, shape )
        g_min = 1.0 / self.r_max
        g_max = 1.0 / self.r_min
        
        return ((g_curr - g_min) / (g_max - g_min)) * gain
    
    def initial_resistances( self, low, high, shape ):
        return np.random.uniform( low, high, shape )
    
    @property
    def _argdefaults( self ):
        return (
                ("learning_rate", PES.learning_rate.default),
                ("pre_synapse", PES.pre_synapse.default),
                )


import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import Reset
from nengo.ensemble import Ensemble
from nengo.node import Node


class SimmPES( Operator ):
    def __init__(
            self,
            pre_filtered,
            error,
            delta,
            learning_rate,
            encoders,
            pos_memristors,
            neg_memristors,
            states=None,
            tag=None
            ):
        super( SimmPES, self ).__init__( tag=tag )
        
        self.learning_rate = learning_rate
        
        self.sets = [ ] + ([ ] if states is None else [ states ])
        self.incs = [ ]
        self.reads = [ pre_filtered, error, encoders ]
        self.updates = [ delta, pos_memristors, neg_memristors ]
    
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
    def delta( self ):
        return self.updates[ 0 ]
    
    @property
    def pos_memristor( self ):
        return self.updates[ 1 ]
    
    @property
    def neg_memristor( self ):
        return self.updates[ 2 ]
    
    def _descstr( self ):
        return "pre=%s, error=%s -> %s" % (self.pre_filtered, self.error, self.delta)
    
    def make_step( self, signals, dt, rng ):
        pre_filtered = signals[ self.pre_filtered ]
        error = signals[ self.error ]
        delta = signals[ self.delta ]
        n_neurons = pre_filtered.shape[ 0 ]
        alpha = -self.learning_rate * dt / n_neurons
        encoders = signals[ self.encoders ]
        pos_memristors = signals[ self.pos_memristor ]
        neg_memristors = signals[ self.neg_memristor ]
        
        def step_simmpes():
            # TODO pass parameters or equations/functions directly
            a = -0.1802
            r_min = 1e2
            r_max = 2.5e8
            g_min = 1.0 / r_max
            g_max = 1.0 / r_min
            gain = 1e5
            error_threshold = 1e-5
            
            # analytical derivative of pulse number
            def monom_deriv( base, exp ):
                return exp * base**(exp - 1)
            
            def find_spikes( input_activities, shape, output_activities=None, invert=False ):
                output_size = shape[ 0 ]
                input_size = shape[ 1 ]
                spiked_pre = np.tile(
                        np.array( np.rint( input_activities ), dtype=bool ), (output_size, 1)
                        )
                spiked_post = np.tile(
                        np.expand_dims(
                                np.array( np.rint( output_activities ), dtype=bool ), axis=1 ), (1, input_size)
                        ) \
                    if output_activities is not None \
                    else np.ones( (1, input_size) )
                
                out = np.logical_and( spiked_pre, spiked_post )
                return out if not invert else np.logical_not( out )
            
            # set update to zero if error is small or adjustments go on for ever
            # if error is small return zero delta
            if np.any( np.absolute( error ) > error_threshold ):
                # calculate the magnitude of the update based on PES learning rule
                local_error = alpha * np.dot( encoders, error )
                pes_delta = np.outer( local_error, pre_filtered )
                
                # some memristors are adjusted erroneously if we don't filter
                spiked_map = find_spikes( pre_filtered, delta.shape, invert=True )
                pes_delta[ spiked_map ] = 0
                
                # set update direction and magnitude (unused with powerlaw memristor equations)
                V = np.sign( pes_delta ) * 1e-1
                
                # update the two memristor pairs separately
                n_pos = ((1 / pos_memristors[ V > 0 ] - 1 / g_max) * g_min)**(1 / a)
                n_neg = ((1 / neg_memristors[ V < 0 ] - 1 / g_max) * g_min)**(1 / a)
                
                delta_pos = -1 * g_max * g_min \
                            * (g_min + g_max * n_pos**a)**(-2) \
                            * g_max * monom_deriv( n_pos, a )
                delta_neg = -1 * g_max * g_min \
                            * (g_min + g_max * n_neg**a)**(-2) \
                            * g_max * monom_deriv( n_neg, a )
                
                pos_memristors[ V > 0 ] += delta_pos
                neg_memristors[ V < 0 ] += delta_neg
                
                delta[ V > 0 ] = delta_pos * gain
                delta[ V < 0 ] = -delta_neg * gain
            else:
                delta[ : ] = np.zeros_like( delta )
        
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
    gain = 1e5
    
    conn = rule.connection
    
    # NB "mpes" is the "mPES()" frontend class
    
    # Create input error signal
    error = Signal( shape=rule.size_in, name="mPES:error" )
    model.add_op( Reset( error ) )
    model.sig[ rule ][ "in" ] = error  # error connection will attach here
    
    # Filter pre-synaptic activities with pre_synapse
    acts = build_or_passthrough( model, mpes.pre_synapse, model.sig[ conn.pre_obj ][ "out" ] )
    
    post = get_post_ens( conn )
    encoders = model.sig[ post ][ "encoders" ][ :, conn.post_slice ]
    
    out_size = encoders.shape[ 0 ]
    in_size = acts.shape[ 0 ]
    
    # initial_conductances_pos = 1 / mpes.initial_resistances( 1e8, 1e8, (out_size, in_size) )
    # initial_conductances_neg = 1 / mpes.initial_resistances( 1e8, 1e8, (out_size, in_size) )
    initial_conductances_pos = mpes.initial_conductances_pos
    initial_conductances_neg = mpes.initial_conductances_neg
    
    pos_memristors = Signal( shape=(out_size, in_size), name="mPES:pos_memristors",
                             initial_value=initial_conductances_pos )
    neg_memristors = Signal( shape=(out_size, in_size), name="mPES:neg_memristors",
                             initial_value=initial_conductances_neg )
    
    model.sig[ rule ][ "pos_memristors" ] = pos_memristors
    model.sig[ rule ][ "neg_memristors" ] = neg_memristors
    
    model.add_op(
            SimmPES(
                    acts,
                    error,
                    model.sig[ rule ][ "delta" ],
                    mpes.learning_rate,
                    encoders,
                    model.sig[ rule ][ "pos_memristors" ],
                    model.sig[ rule ][ "neg_memristors" ]
                    )
            )
    
    # expose these for probes
    model.sig[ rule ][ "error" ] = error
    model.sig[ rule ][ "activities" ] = acts
    model.sig[ rule ][ "pos_memristors" ] = pos_memristors
    model.sig[ rule ][ "neg_memristors" ] = neg_memristors
