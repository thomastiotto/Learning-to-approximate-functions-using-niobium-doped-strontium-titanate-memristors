import warnings

from nengo.params import Default, NumberParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.learning_rules import LearningRuleType
from nengo.learning_rules import PES


class mPES( LearningRuleType ):
    modifies = "weights"
    probeable = ("error", "activities", "delta")
    
    learning_rate = NumberParam( "learning_rate", low=0, readonly=True, default=1e-4 )
    pre_synapse = SynapseParam( "pre_synapse", default=Lowpass( tau=0.005 ), readonly=True )
    r_max = NumberParam( "r_max", readonly=True, default=2.5e8 )
    r_min = NumberParam( "r_min", readonly=True, default=1e2 )
    
    def __init__( self, learning_rate=Default, pre_synapse=Default, r_max=Default, r_min=Default ):
        super().__init__( learning_rate, size_in="post_state" )
        if learning_rate is not Default and learning_rate >= 1.0:
            warnings.warn(
                    "This learning rate is very high, and can result "
                    "in floating point errors from too much current."
                    )
        
        self.pre_synapse = pre_synapse
        self.r_max = r_max
        self.r_min = r_min
    
    def normalized_conductance( self, R ):
        epsilon = np.finfo( float ).eps
        gain = 1e5
        
        g_curr = 1.0 / R
        g_min = 1.0 / self.r_max
        g_max = 1.0 / self.r_min
        
        return gain * (((g_curr - g_min) / (g_max - g_min)) + epsilon)
    
    # TODO return normalized conductances
    @staticmethod
    def initial_normalized_conductances( low, high, shape, r_max=2.5e8, r_min=1e2 ):
        epsilon = np.finfo( float ).eps
        gain = 1e5
        
        g_curr = 1.0 / np.random.uniform( low, high, shape )
        g_min = 1.0 / r_max
        g_max = 1.0 / r_min
        
        return gain * (((g_curr - g_min) / (g_max - g_min)) + epsilon)
    
    @staticmethod
    def initial_resistances( low, high, shape ):
        return np.random.uniform( low, high, shape )
    
    @property
    def _argdefaults( self ):
        return (
                ("learning_rate", PES.learning_rate.default),
                ("pre_synapse", PES.pre_synapse.default),
                )


import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble
from nengo.exceptions import BuildError
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
            # TODO pass parameters or equations/functions directly
            a = -0.128
            # TODO learning works but reverse bias has basically no effect
            c = -1e-3
            r_min = 1e2
            r_max = 2.5e8
            g_min = 1.0 / r_max
            g_max = 1.0 / r_min
            r_3 = 1e9
            gain = 1e5
            epsilon = np.finfo( float ).eps
            dtt = dt
            
            # calculate the magnitude of the update based on PES learning rule
            local_error = alpha * np.dot( encoders, error )
            pes_delta = np.outer( local_error, pre_filtered )
            
            # find neurons that spiked and set the other updates to zero
            # spiked_map = find_spikes( pre_filtered, pes_delta.shape )
            # np.logical_and( pes_delta, spiked_map, out=pes_delta )
            
            # analytical derivative of pulse number
            def deriv( n, a ):
                return a * n**(a - 1)
            
            def conductance2resistance( G ):
                g_clean = (G / gain) - epsilon
                g_unnorm = g_clean * (g_max - g_min) + g_min
                
                return 1.0 / g_unnorm
            
            def resistance2conductance( R ):
                g_curr = 1.0 / R
                g_norm = (g_curr - g_min) / (g_max - g_min)
                
                return gain * (g_norm + epsilon)
            
            # convert connection weights to un-normalized resistance
            weights_res = conductance2resistance( weights )
            # set update direction and magnitude (unused with powerlaw memristor equations)
            V = np.sign( pes_delta ) * 1e-1
            # use the correct equation for the bidirectional powerlaw memristor update
            # I am using forward difference 1st order approximation to calculate the update delta
            try:
                pos_update = r_max * deriv( (((weights_res[ V > 0 ] - r_min) / r_max)**(1 / a)), a )
                neg_update = -r_3 * deriv( (((r_3 - weights_res[ V < 0 ]) / r_3)**(1 / c)), c )
                delta[ V > 0 ] = resistance2conductance( weights_res[ V > 0 ] + pos_update ) - weights[ V > 0 ]
                delta[ V < 0 ] = weights[ V < 0 ] - resistance2conductance( weights_res[ V < 0 ] + neg_update )
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
