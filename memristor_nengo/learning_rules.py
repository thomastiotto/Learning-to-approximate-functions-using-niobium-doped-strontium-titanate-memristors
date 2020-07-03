import warnings
import numpy as np

from nengo.params import Default, NumberParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.learning_rules import LearningRuleType

from nengo.builder.learning_rules import get_post_ens, build_or_passthrough
from nengo.builder import Builder, Operator


class mPES( LearningRuleType ):
    modifies = "weights"
    probeable = ("error", "activities", "delta", "pos_memristors", "neg_memristors")
    
    learning_rate = NumberParam( "learning_rate", low=0, readonly=True, default=1e-4 )
    pre_synapse = SynapseParam( "pre_synapse", default=Lowpass( tau=0.005 ), readonly=True )
    r_max = NumberParam( "r_max", readonly=True, default=2.5e8 )
    r_min = NumberParam( "r_min", readonly=True, default=1e2 )
    
    def __init__( self, learning_rate=Default, pre_synapse=Default, r_max=Default, r_min=Default, noisy=False,
                  seed=None ):
        super().__init__( learning_rate, size_in="post_state" )
        if learning_rate is not Default and learning_rate >= 1.0:
            warnings.warn(
                    "This learning rate is very high, and can result "
                    "in floating point errors from too much current."
                    )
        
        self.pre_synapse = pre_synapse
        self.r_max = r_max
        self.r_min = r_min
        self.noise_percentage = 0 if not noisy else noisy
        
        np.random.seed( seed )
    
    def initial_resistances( self, low, high, shape ):
        return np.random.uniform( low, high, shape )
    
    @property
    def _argdefaults( self ):
        return (
                ("learning_rate", mPES.learning_rate.default),
                ("pre_synapse", mPES.pre_synapse.default),
                )


class SimmPES( Operator ):
    def __init__(
            self,
            pre_filtered,
            error,
            learning_rate,
            encoders,
            pos_memristors,
            neg_memristors,
            weights,
            noise_percentage,
            states=None,
            tag=None
            ):
        super( SimmPES, self ).__init__( tag=tag )
        
        self.learning_rate = learning_rate
        self.noise_percentage = noise_percentage
        
        self.sets = [ ] + ([ ] if states is None else [ states ])
        self.incs = [ ]
        self.reads = [ pre_filtered, error, encoders ]
        self.updates = [ weights, pos_memristors, neg_memristors ]
    
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
        return self.updates[ 0 ]
    
    @property
    def pos_memristor( self ):
        return self.updates[ 1 ]
    
    @property
    def neg_memristor( self ):
        return self.updates[ 2 ]
    
    def _descstr( self ):
        return "pre=%s, error=%s -> %s" % (self.pre_filtered, self.error, self.weights)
    
    def make_step( self, signals, dt, rng ):
        pre_filtered = signals[ self.pre_filtered ]
        n_neurons = pre_filtered.shape[ 0 ]
        error = signals[ self.error ]
        encoders = signals[ self.encoders ]
        
        pos_memristors = signals[ self.pos_memristor ]
        neg_memristors = signals[ self.neg_memristor ]
        weights = signals[ self.weights ]
        gain = 1e6 / n_neurons
        noise_percentage = self.noise_percentage
        
        def step_simmpes():
            # TODO pass parameters or equations/functions directly
            a = -0.1
            r_min = 1e2
            r_max = 2.5e8
            g_min = 1.0 / r_max
            g_max = 1.0 / r_min
            error_threshold = 1e-5
            
            def resistance2conductance( R ):
                g_curr = 1.0 / R
                g_norm = (g_curr - g_min) / (g_max - g_min)
                
                return g_norm * gain
            
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
                local_error = -np.dot( encoders, error )
                pes_delta = np.outer( local_error, pre_filtered )
                
                # some memristors are adjusted erroneously if we don't filter
                spiked_map = find_spikes( pre_filtered, weights.shape, invert=True )
                pes_delta[ spiked_map ] = 0
                
                # set update direction and magnitude (unused with powerlaw memristor equations)
                V = np.sign( pes_delta ) * 1e-1
                
                if noise_percentage > 0:
                    # generate random noise on the parameters
                    r_min_noisy = np.random.normal( r_min, r_min * noise_percentage, V.shape )
                    r_max_noisy = np.random.normal( r_max, r_max * noise_percentage, V.shape )
                    a_noisy = np.random.normal( a, np.abs( a ) * noise_percentage, V.shape )
                    
                    # update the two memristor pairs separately
                    n_pos = ((pos_memristors[ V > 0 ] - r_min_noisy[ V > 0 ]) / r_max_noisy[ V > 0 ])**(
                            1 / a_noisy[ V > 0 ])
                    n_neg = ((neg_memristors[ V < 0 ] - r_min_noisy[ V < 0 ]) / r_max_noisy[ V < 0 ])**(
                            1 / a_noisy[ V < 0 ])
                    
                    pos_memristors[ V > 0 ] = r_min_noisy[ V > 0 ] + r_max_noisy[ V > 0 ] * (n_pos + 1)**a_noisy[
                        V > 0 ]
                    neg_memristors[ V < 0 ] = r_min_noisy[ V < 0 ] + r_max_noisy[ V < 0 ] * (n_neg + 1)**a_noisy[
                        V < 0 ]
                else:
                    # update the two memristor pairs separately
                    n_pos = ((pos_memristors[ V > 0 ] - r_min) /
                             r_max)**(1 / a)
                    n_neg = ((neg_memristors[ V < 0 ] - r_min) /
                             r_max)**(1 / a)
                    
                    pos_memristors[ V > 0 ] = r_min + r_max * (n_pos + 1)**a
                    neg_memristors[ V < 0 ] = r_min + r_max * (n_neg + 1)**a
                
                weights[ : ] = resistance2conductance( pos_memristors[ : ] ) \
                               - resistance2conductance( neg_memristors[ : ] )
        
        return step_simmpes


@Builder.register( mPES )
def build_mpes( model, mpes, rule ):
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
    
    pos_memristors = Signal( shape=(out_size, in_size), name="mPES:pos_memristors",
                             initial_value=mpes.initial_resistances( 1e8 - 1e8 * mpes.noise_percentage,
                                                                     1.1e8 + 1e8 * mpes.noise_percentage,
                                                                     (out_size, in_size) ) )
    neg_memristors = Signal( shape=(out_size, in_size), name="mPES:neg_memristors",
                             initial_value=mpes.initial_resistances( 1e8 - 1e8 * mpes.noise_percentage,
                                                                     1.1e8 + 1e8 * mpes.noise_percentage,
                                                                     (out_size, in_size) ) )
    
    model.sig[ conn ][ "pos_memristors" ] = pos_memristors
    model.sig[ conn ][ "neg_memristors" ] = neg_memristors
    
    model.add_op(
            SimmPES(
                    acts,
                    error,
                    mpes.learning_rate,
                    encoders,
                    model.sig[ conn ][ "pos_memristors" ],
                    model.sig[ conn ][ "neg_memristors" ],
                    model.sig[ conn ][ "weights" ],
                    mpes.noise_percentage
                    )
            )
    
    # expose these for probes
    model.sig[ rule ][ "error" ] = error
    model.sig[ rule ][ "activities" ] = acts
    model.sig[ rule ][ "pos_memristors" ] = pos_memristors
    model.sig[ rule ][ "neg_memristors" ] = neg_memristors


################ NENGO DL #####################

import tensorflow as tf
from nengo.builder import Signal
from nengo.builder.operator import Reset, DotInc, Copy

from nengo_dl.builder import Builder, OpBuilder, NengoBuilder


@NengoBuilder.register( mPES )
def build_pes( model, mpes, rule ):
    conn = rule.connection
    
    # Create input error signal
    error = Signal( shape=(rule.size_in,), name="PES:error" )
    model.add_op( Reset( error ) )
    model.sig[ rule ][ "in" ] = error  # error connection will attach here
    
    acts = build_or_passthrough( model, mpes.pre_synapse, model.sig[ conn.pre_obj ][ "out" ] )
    
    post = get_post_ens( conn )
    encoders = model.sig[ post ][ "encoders" ]
    
    out_size = encoders.shape[ 0 ]
    in_size = acts.shape[ 0 ]
    
    pos_memristors = Signal( shape=(out_size, in_size), name="mPES:pos_memristors",
                             initial_value=mpes.initial_resistances( 1e8 - 1e8 * mpes.noise_percentage,
                                                                     1.1e8 + 1e8 * mpes.noise_percentage,
                                                                     (out_size, in_size) ) )
    neg_memristors = Signal( shape=(out_size, in_size), name="mPES:neg_memristors",
                             initial_value=mpes.initial_resistances( 1e8 - 1e8 * mpes.noise_percentage,
                                                                     1.1e8 + 1e8 * mpes.noise_percentage,
                                                                     (out_size, in_size) ) )
    
    model.sig[ conn ][ "pos_memristors" ] = pos_memristors
    model.sig[ conn ][ "neg_memristors" ] = neg_memristors
    
    if conn.post_obj is not conn.post:
        # in order to avoid slicing encoders along an axis > 0, we pad
        # `error` out to the full base dimensionality and then do the
        # dotinc with the full encoder matrix
        padded_error = Signal( shape=(encoders.shape[ 1 ],) )
        model.add_op( Copy( error, padded_error, dst_slice=conn.post_slice ) )
    else:
        padded_error = error
    
    # error = dot(encoders, error)
    local_error = Signal( shape=(post.n_neurons,) )
    model.add_op( Reset( local_error ) )
    model.add_op( DotInc( encoders, padded_error, local_error, tag="PES:encode" ) )
    
    model.operators.append(
            SimmPES(
                    acts,
                    local_error,
                    mpes.learning_rate,
                    encoders,
                    model.sig[ conn ][ "pos_memristors" ],
                    model.sig[ conn ][ "neg_memristors" ],
                    model.sig[ conn ][ "weights" ],
                    mpes.noise_percentage
                    )
            )
    
    # expose these for probes
    model.sig[ rule ][ "error" ] = error
    model.sig[ rule ][ "activities" ] = acts
    model.sig[ rule ][ "pos_memristors" ] = pos_memristors
    model.sig[ rule ][ "neg_memristors" ] = neg_memristors


@Builder.register( SimmPES )
class SimPESBuilder( OpBuilder ):
    """Build a group of `~nengo.builder.learning_rules.SimPES` operators."""
    
    def __init__( self, ops, signals, config ):
        super().__init__( ops, signals, config )
        
        self.output_size = ops[ 0 ].weights.shape[ 0 ]
        self.input_size = ops[ 0 ].weights.shape[ 1 ]
        
        self.error_data = signals.combine( [ op.error for op in ops ] )
        self.error_data = self.error_data.reshape( (len( ops ), ops[ 0 ].error.shape[ 0 ], 1) )
        
        self.pre_data = signals.combine( [ op.pre_filtered for op in ops ] )
        self.pre_data = self.pre_data.reshape( (len( ops ), 1, ops[ 0 ].pre_filtered.shape[ 0 ]) )
        
        self.output_data = signals.combine( [ op.weights for op in ops ] )
    
    def build_step( self, signals ):
        pre_filtered = signals.gather( self.pre_data )
        local_error = signals.gather( self.error_data )
        
        def find_spikes( input_activities, output_size, invert=False ):
            spiked_pre = tf.cast( tf.tile( tf.math.rint( input_activities ), [ 1, 1, output_size, 1 ] ),
                                  tf.bool )
            
            out = spiked_pre
            if invert:
                out = tf.math.logical_not( out )
            
            return tf.cast( out, tf.float32 )
        
        pes_delta = -local_error * pre_filtered
        
        spiked_map = find_spikes( pre_filtered, self.output_size, invert=True )
        pes_delta = pes_delta * spiked_map
        # pes_delta.set_shape( [ 1, 1, self.output_size, self.input_size ] )
        V = tf.sign( pes_delta ) * 1e-1
        
        signals.scatter( self.output_data, pes_delta )


@staticmethod
def mergeable( x, y ):
    # pre inputs must have the same dimensionality so that we can broadcast
    # them when computing the outer product.
    # the error signals also have to have the same shape.
    return (
            x.pre_filtered.shape[ 0 ] == y.pre_filtered.shape[ 0 ]
            and x.error.shape[ 0 ] == y.error.shape[ 0 ]
    )
