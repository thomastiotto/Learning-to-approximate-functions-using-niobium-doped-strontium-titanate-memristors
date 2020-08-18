import warnings
import numpy as np

from nengo.params import Default, NumberParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.learning_rules import LearningRuleType

from nengo.builder.learning_rules import get_post_ens, build_or_passthrough
from nengo.builder import Operator


class mPES( LearningRuleType ):
    modifies = "weights"
    probeable = ("error", "activities", "delta", "pos_memristors", "neg_memristors")
    
    learning_rate = NumberParam( "learning_rate", low=0, readonly=True, default=1e-4 )
    pre_synapse = SynapseParam( "pre_synapse", default=Lowpass( tau=0.005 ), readonly=True )
    r_max = NumberParam( "r_max", readonly=True, default=2.3e8 )
    r_min = NumberParam( "r_min", readonly=True, default=200 )
    exponent = NumberParam( "exponent", readonly=True, default=-0.146 )
    
    def __init__( self,
                  learning_rate=Default,
                  pre_synapse=Default,
                  r_max=Default,
                  r_min=Default,
                  exponent=Default,
                  noisy=False,
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
        self.exponent = exponent
        self.noise_percentage = 0 if not noisy else noisy
        
        np.random.seed( seed )
        tf.random.set_seed( seed )
    
    def initial_resistances( self, low, high, shape ):
        return np.random.uniform( low, high, shape )
    
    @property
    def _argdefaults( self ):
        return (
                ("learning_rate", mPES.learning_rate.default),
                ("pre_synapse", mPES.pre_synapse.default),
                ("r_max", mPES.r_max.default),
                ("r_min", mPES.r_min.default),
                ("exponent", mPES.exponent.default),
                )


class SimmPES( Operator ):
    def __init__(
            self,
            pre_filtered,
            error,
            learning_rate,
            pos_memristors,
            neg_memristors,
            weights,
            noise_percentage,
            r_max,
            r_min,
            exponent,
            states=None,
            tag=None
            ):
        super( SimmPES, self ).__init__( tag=tag )
        
        self.pre_n_neurons = weights.shape[ 1 ]
        self.post_n_neurons = weights.shape[ 0 ]
        self.learning_rate = learning_rate
        self.noise_percentage = noise_percentage
        self.gain = 1e6 / self.pre_n_neurons
        self.error_threshold = 1e-5
        self.r_max = r_max
        self.r_min = r_min
        self.exponent = exponent
        
        self.sets = [ ] + ([ ] if states is None else [ states ])
        self.incs = [ ]
        self.reads = [ pre_filtered, error ]
        self.updates = [ weights, pos_memristors, neg_memristors ]
    
    @property
    def pre_filtered( self ):
        return self.reads[ 0 ]
    
    @property
    def error( self ):
        return self.reads[ 1 ]
    
    @property
    def weights( self ):
        return self.updates[ 0 ]
    
    @property
    def pos_memristors( self ):
        return self.updates[ 1 ]
    
    @property
    def neg_memristors( self ):
        return self.updates[ 2 ]
    
    def _descstr( self ):
        return "pre=%s, error=%s -> %s" % (self.pre_filtered, self.error, self.weights)
    
    def make_step( self, signals, dt, rng ):
        pre_filtered = signals[ self.pre_filtered ]
        local_error = signals[ self.error ]
        
        pos_memristors = signals[ self.pos_memristors ]
        neg_memristors = signals[ self.neg_memristors ]
        weights = signals[ self.weights ]
        
        gain = self.gain
        noise_percentage = self.noise_percentage
        error_threshold = self.error_threshold
        r_min = self.r_min
        r_max = self.r_max
        exponent = self.exponent
        g_min = 1.0 / r_max
        g_max = 1.0 / r_min
        
        def step_simmpes():
            
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
            if np.any( np.absolute( local_error ) > error_threshold ):
                # calculate the magnitude of the update based on PES learning rule
                # local_error = -np.dot( encoders, error )
                # I can use NengoDL build function like this, as dot(encoders, error) has been done there already
                # i.e., error already contains the PES local error
                pes_delta = np.outer( -local_error, pre_filtered )
                
                # some memristors are adjusted erroneously if we don't filter
                spiked_map = find_spikes( pre_filtered, weights.shape, invert=True )
                pes_delta[ spiked_map ] = 0
                
                # set update direction and magnitude (unused with powerlaw memristor equations)
                V = np.sign( pes_delta ) * 1e-1
                
                if noise_percentage > 0:
                    # generate random noise on the parameters
                    exponent_noisy = np.random.normal( exponent, np.abs( exponent ) * noise_percentage, V.shape )
                    r_min_noisy = np.random.normal( r_min, r_min * noise_percentage, V.shape )
                    r_max_noisy = np.random.normal( r_max, r_max * noise_percentage, V.shape )
                    
                    # update the two memristor pairs separately
                    pos_n = ((pos_memristors[ V > 0 ] - r_min_noisy[ V > 0 ]) / r_max_noisy[ V > 0 ])**(
                            1 / exponent_noisy[ V > 0 ])
                    pos_memristors[ V > 0 ] = r_min_noisy[ V > 0 ] + r_max_noisy[ V > 0 ] * (pos_n + 1)**exponent_noisy[
                        V > 0 ]
                    
                    neg_n = ((neg_memristors[ V < 0 ] - r_min_noisy[ V < 0 ]) / r_max_noisy[ V < 0 ])**(
                            1 / exponent_noisy[ V < 0 ])
                    neg_memristors[ V < 0 ] = r_min_noisy[ V < 0 ] + r_max_noisy[ V < 0 ] * (neg_n + 1)**exponent_noisy[
                        V < 0 ]
                else:
                    # updating the two memristor pairs separately
                    pos_n = ((pos_memristors[ V > 0 ] - r_min) / r_max)**(1 / exponent)
                    pos_update = r_min + r_max * (pos_n + 1)**exponent
                    pos_memristors[ V > 0 ] = pos_update
                    
                    neg_n = ((neg_memristors[ V < 0 ] - r_min) / r_max)**(1 / exponent)
                    neg_update = r_min + r_max * (neg_n + 1)**exponent
                    neg_memristors[ V < 0 ] = neg_update
                
                weights[ : ] = resistance2conductance( pos_memristors[ : ] ) \
                               - resistance2conductance( neg_memristors[ : ] )
        
        return step_simmpes


################ NENGO DL #####################

import tensorflow as tf
from nengo.builder import Signal
from nengo.builder.operator import Reset, DotInc, Copy

from nengo_dl.builder import Builder, OpBuilder, NengoBuilder
from nengo.builder import Builder as NengoCoreBuilder


@NengoBuilder.register( mPES )
@NengoCoreBuilder.register( mPES )
def build_mpes( model, mpes, rule ):
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
        # comes into effect when slicing post connection
        padded_error = Signal( shape=(encoders.shape[ 1 ],) )
        model.add_op( Copy( error, padded_error, dst_slice=conn.post_slice ) )
    else:
        padded_error = error
    
    # error = dot(encoders, error)
    local_error = Signal( shape=(post.n_neurons,) )
    model.add_op( Reset( local_error ) )
    model.add_op( DotInc( encoders, padded_error, local_error, tag="PES:encode" ) )
    
    model.operators.append(
            SimmPES( acts,
                     local_error,
                     mpes.learning_rate,
                     model.sig[ conn ][ "pos_memristors" ],
                     model.sig[ conn ][ "neg_memristors" ],
                     model.sig[ conn ][ "weights" ],
                     mpes.noise_percentage,
                     mpes.r_max,
                     mpes.r_min,
                     mpes.exponent )
            )
    
    # expose these for probes
    model.sig[ rule ][ "error" ] = error
    model.sig[ rule ][ "activities" ] = acts
    model.sig[ rule ][ "pos_memristors" ] = pos_memristors
    model.sig[ rule ][ "neg_memristors" ] = neg_memristors


@Builder.register( SimmPES )
class SimmPESBuilder( OpBuilder ):
    """Build exponent group of `~nengo.builder.learning_rules.SimmPES` operators."""
    
    def build_pre( self, signals, config ):
        super().build_pre( signals, config )
        
        self.output_size = self.ops[ 0 ].weights.shape[ 0 ]
        self.input_size = self.ops[ 0 ].weights.shape[ 1 ]
        
        self.error_data = signals.combine( [ op.error for op in self.ops ] )
        self.error_data = self.error_data.reshape( (len( self.ops ), self.ops[ 0 ].error.shape[ 0 ], 1) )
        
        self.pre_data = signals.combine( [ op.pre_filtered for op in self.ops ] )
        self.pre_data = self.pre_data.reshape( (len( self.ops ), 1, self.ops[ 0 ].pre_filtered.shape[ 0 ]) )
        
        self.pos_memristors = signals.combine( [ op.pos_memristors for op in self.ops ] )
        self.pos_memristors = self.pos_memristors.reshape(
                (len( self.ops ), self.ops[ 0 ].pos_memristors.shape[ 0 ], self.ops[ 0 ].pos_memristors.shape[ 1 ])
                )
        
        self.neg_memristors = signals.combine( [ op.neg_memristors for op in self.ops ] )
        self.neg_memristors = self.neg_memristors.reshape(
                (len( self.ops ), self.ops[ 0 ].neg_memristors.shape[ 0 ], self.ops[ 0 ].neg_memristors.shape[ 1 ])
                )
        
        self.output_data = signals.combine( [ op.weights for op in self.ops ] )
        
        self.gain = signals.op_constant( self.ops,
                                         [ 1 for _ in self.ops ],
                                         "gain",
                                         signals.dtype,
                                         shape=(1, -1, 1, 1) )
        self.noise_percentage = signals.op_constant( self.ops,
                                                     [ 1 for _ in self.ops ],
                                                     "noise_percentage",
                                                     signals.dtype,
                                                     shape=(1, -1, 1, 1) )
        self.r_max = signals.op_constant( self.ops,
                                          [ 1 for _ in self.ops ],
                                          "r_max",
                                          signals.dtype,
                                          shape=(1, -1, 1, 1) )
        self.r_min = signals.op_constant( self.ops,
                                          [ 1 for _ in self.ops ],
                                          "r_min",
                                          signals.dtype,
                                          shape=(1, -1, 1, 1) )
        self.exponent = signals.op_constant( self.ops,
                                             [ 1 for _ in self.ops ],
                                             "exponent",
                                             signals.dtype,
                                             shape=(1, -1, 1, 1) )
        self.error_threshold = signals.op_constant( self.ops,
                                                    [ 1 for _ in self.ops ],
                                                    "error_threshold",
                                                    signals.dtype,
                                                    shape=(1, -1, 1, 1) )
        self.post_n_neurons = signals.op_constant( self.ops,
                                                   [ 1 for _ in self.ops ],
                                                   "post_n_neurons",
                                                   signals.dtype,
                                                   shape=(1, -1, 1, 1) )
        self.g_min = 1.0 / self.r_max
        self.g_max = 1.0 / self.r_min
    
    def build_step( self, signals ):
        pre_filtered = signals.gather( self.pre_data )
        local_error = signals.gather( self.error_data )
        pos_memristors = signals.gather( self.pos_memristors )
        neg_memristors = signals.gather( self.neg_memristors )
        
        def resistance2conductance( R ):
            g_curr = 1.0 / R
            g_norm = (g_curr - self.g_min) / (self.g_max - self.g_min)
            
            return g_norm * self.gain
        
        def find_spikes( input_activities, output_size, invert=False ):
            spiked_pre = tf.cast(
                    tf.tile( tf.math.rint( input_activities ), [ 1, 1, output_size, 1 ] ),
                    tf.bool )
            
            out = spiked_pre
            if invert:
                out = tf.math.logical_not( out )
            
            return tf.cast( out, tf.float32 )
        
        def if_noise_greater_than_zero( pos_memristors, neg_memristors ):
            # generate noisy parameters
            exponent_noisy = tf.random.normal( V.shape, self.exponent, tf.abs( self.exponent ) * self.noise_percentage )
            r_min_noisy = tf.random.normal( V.shape, self.r_min, self.r_min * self.noise_percentage )
            r_max_noisy = tf.random.normal( V.shape, self.r_max, self.r_max * self.noise_percentage )
            
            # positive memristors update
            pos_mask = tf.greater( V, 0 )
            pos_indices = tf.where( pos_mask )
            pos_n = (
                            (tf.boolean_mask( pos_memristors, pos_mask ) - tf.boolean_mask( r_min_noisy, pos_mask ))
                            / tf.boolean_mask( r_max_noisy, pos_mask )
                    )**(1 / tf.boolean_mask( exponent_noisy, pos_mask ))
            pos_update = tf.boolean_mask( r_min_noisy, pos_mask ) \
                         + tf.boolean_mask( r_max_noisy, pos_mask ) \
                         * (pos_n + 1)**tf.boolean_mask( exponent_noisy, pos_mask )
            pos_memristors = tf.tensor_scatter_nd_update( pos_memristors, pos_indices, pos_update )
            
            # negative memristors update
            neg_mask = tf.less( V, 0 )
            neg_indices = tf.where( neg_mask )
            neg_n = (
                            (tf.boolean_mask( neg_memristors, neg_mask ) - tf.boolean_mask( r_min_noisy, neg_mask ))
                            / tf.boolean_mask( r_max_noisy, neg_mask )
                    )**(1 / tf.boolean_mask( exponent_noisy, neg_mask ))
            neg_update = tf.boolean_mask( r_min_noisy, neg_mask ) \
                         + tf.boolean_mask( r_max_noisy, neg_mask ) \
                         * (neg_n + 1)**tf.boolean_mask( exponent_noisy, neg_mask )
            neg_memristors = tf.tensor_scatter_nd_update( neg_memristors, neg_indices, neg_update )
            
            return pos_memristors, neg_memristors
        
        def if_noise_is_zero( pos_memristors, neg_memristors ):
            # positive memristors update
            pos_mask = tf.greater( V, 0 )
            pos_indices = tf.where( pos_mask )
            pos_n = ((tf.boolean_mask( pos_memristors, pos_mask ) - self.r_min) / self.r_max)**(1 / self.exponent)
            pos_update = self.r_min + self.r_max * (pos_n + 1)**self.exponent
            pos_memristors = tf.tensor_scatter_nd_update( pos_memristors, pos_indices, pos_update )
            
            # negative memristors update
            neg_mask = tf.less( V, 0 )
            neg_indices = tf.where( neg_mask )
            neg_n = ((tf.boolean_mask( neg_memristors, neg_mask ) - self.r_min) / self.r_max)**(1 / self.exponent)
            neg_update = self.r_min + self.r_max * (neg_n + 1)**self.exponent
            neg_memristors = tf.tensor_scatter_nd_update( neg_memristors, neg_indices, neg_update )
            
            return pos_memristors, neg_memristors
        
        pes_delta = -local_error * pre_filtered
        
        spiked_map = find_spikes( pre_filtered, self.post_n_neurons )
        pes_delta = pes_delta * spiked_map
        
        V = tf.sign( pes_delta ) * 1e-1
        
        # SECOND thing, called when error is over threshold
        # if added noise is zero then executes noiseless memristors branch
        # if added noise is greater than zero then calls the noisy memristors branch
        def if_error_over_threshold( pos_memristors, neg_memristors ):
            pos_memristors, neg_memristors = tf.cond( tf.greater( self.noise_percentage, 0 ),
                                                      true_fn=lambda: if_noise_greater_than_zero( pos_memristors,
                                                                                                  neg_memristors ),
                                                      false_fn=lambda: if_noise_is_zero( pos_memristors,
                                                                                         neg_memristors )
                                                      )
            
            return pos_memristors, neg_memristors
        
        # FIRST thing, check if the error is greater than the threshold
        # if any errors is above threshold then pass decision to next tf.cond()
        # if all errors are below threshold then do nothing
        pos_memristors, neg_memristors = tf.cond(
                tf.reduce_any( tf.greater( tf.abs( local_error ), self.error_threshold ) ),
                true_fn=lambda: if_error_over_threshold( pos_memristors,
                                                         neg_memristors ),
                false_fn=lambda: (
                        tf.identity( pos_memristors ),
                        tf.identity( neg_memristors ))
                )
        
        # update the memristor values
        signals.scatter(
                self.pos_memristors.reshape( (self.pos_memristors.shape[ -2 ], self.pos_memristors.shape[ -1 ]) ),
                pos_memristors )
        signals.scatter(
                self.neg_memristors.reshape( (self.neg_memristors.shape[ -2 ], self.neg_memristors.shape[ -1 ]) ),
                neg_memristors )
        
        new_weights = resistance2conductance( pos_memristors ) - resistance2conductance( neg_memristors )
        
        signals.scatter( self.output_data, new_weights )
    
    @staticmethod
    def mergeable( x, y ):
        # pre inputs must have the same dimensionality so that we can broadcast
        # them when computing the outer product.
        # the error signals also have to have the same shape.
        return (
                x.pre_filtered.shape[ 0 ] == y.pre_filtered.shape[ 0 ]
                and x.error.shape[ 0 ] == y.error.shape[ 0 ]
        )
