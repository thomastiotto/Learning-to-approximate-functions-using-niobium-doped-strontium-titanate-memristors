import warnings

import numpy as np

from nengo.builder import Operator
from nengo.builder.learning_rules import build_or_passthrough, get_post_ens, get_pre_ens
from nengo.learning_rules import LearningRuleType
from nengo.params import Default, NumberParam
from nengo.synapses import Lowpass, SynapseParam

from scipy.stats import truncnorm


def get_truncated_normal( mean, sd, low, upp, out_size, in_size ):
    try:
        return truncnorm( (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd ) \
            .rvs( out_size * in_size ) \
            .reshape( (out_size, in_size) )
    except ZeroDivisionError:
        return np.full( (out_size, in_size), mean )


def resistance2conductance( R, r_min, r_max ):
    g_min = 1.0 / r_max
    g_max = 1.0 / r_min
    g_curr = 1.0 / R
    
    g_norm = (g_curr - g_min) / (g_max - g_min)
    
    return g_norm


def initialise_memristors( rule, in_size, out_size ):
    with warnings.catch_warnings():
        warnings.simplefilter( "ignore" )
        np.random.seed( rule.seed )
        r_min_noisy = get_truncated_normal( rule.r_min, rule.r_min * rule.noise_percentage[ 0 ],
                                            0, np.inf, out_size, in_size )
        np.random.seed( rule.seed )
        r_max_noisy = get_truncated_normal( rule.r_max, rule.r_max * rule.noise_percentage[ 1 ],
                                            np.max( r_min_noisy ), np.inf, out_size, in_size )
    np.random.seed( rule.seed )
    exponent_noisy = np.random.normal( rule.exponent, np.abs( rule.exponent ) * rule.noise_percentage[ 2 ],
                                       (out_size, in_size) )
    np.random.seed( rule.seed )
    pos_mem_initial = np.random.normal( 1e8, 1e8 * rule.noise_percentage[ 3 ],
                                        (out_size, in_size) )
    np.random.seed( rule.seed + 1 ) if rule.seed else np.random.seed( rule.seed )
    neg_mem_initial = np.random.normal( 1e8, 1e8 * rule.noise_percentage[ 3 ],
                                        (out_size, in_size) )
    
    pos_memristors = Signal( shape=(out_size, in_size), name=f"{rule}:pos_memristors",
                             initial_value=pos_mem_initial )
    neg_memristors = Signal( shape=(out_size, in_size), name=f"{rule}:neg_memristors",
                             initial_value=neg_mem_initial )
    
    return pos_memristors, neg_memristors, r_min_noisy, r_max_noisy, exponent_noisy


def clip_memristor_values( V, pos_memristors, neg_memristors, r_max, r_min ):
    pos_memristors[ V > 0 ] = np.where( pos_memristors[ V > 0 ] > r_max[ V > 0 ],
                                        r_max[ V > 0 ],
                                        pos_memristors[ V > 0 ] )
    pos_memristors[ V > 0 ] = np.where( pos_memristors[ V > 0 ] < r_min[ V > 0 ],
                                        r_min[ V > 0 ],
                                        pos_memristors[ V > 0 ] )
    neg_memristors[ V < 0 ] = np.where( neg_memristors[ V < 0 ] > r_max[ V < 0 ],
                                        r_max[ V < 0 ],
                                        neg_memristors[ V < 0 ] )
    neg_memristors[ V < 0 ] = np.where( neg_memristors[ V < 0 ] < r_min[ V < 0 ],
                                        r_min[ V < 0 ],
                                        neg_memristors[ V < 0 ] )


def update_memristors( V, pos_memristors, neg_memristors, r_max, r_min, exponent ):
    pos_n = np.power( (pos_memristors[ V > 0 ] - r_min[ V > 0 ]) / r_max[ V > 0 ],
                      1 / exponent[ V > 0 ] )
    pos_memristors[ V > 0 ] = r_min[ V > 0 ] + r_max[ V > 0 ] * np.power( pos_n + 1, exponent[ V > 0 ] )
    
    neg_n = np.power( (neg_memristors[ V < 0 ] - r_min[ V < 0 ]) / r_max[ V < 0 ], 1 / exponent[ V < 0 ] )
    neg_memristors[ V < 0 ] = r_min[ V < 0 ] + r_max[ V < 0 ] * np.power( neg_n + 1, exponent[ V < 0 ] )


def update_weights( V, weights, pos_memristors, neg_memristors, r_max, r_min, gain ):
    weights[ V > 0 ] = gain * \
                       (resistance2conductance( pos_memristors[ V > 0 ], r_min[ V > 0 ],
                                                r_max[ V > 0 ] )
                        - resistance2conductance( neg_memristors[ V > 0 ], r_min[ V > 0 ],
                                                  r_max[ V > 0 ] ))
    weights[ V < 0 ] = gain * \
                       (resistance2conductance( pos_memristors[ V < 0 ], r_min[ V < 0 ],
                                                r_max[ V < 0 ] )
                        - resistance2conductance( neg_memristors[ V < 0 ], r_min[ V < 0 ],
                                                  r_max[ V < 0 ] ))


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


class mOja( LearningRuleType ):
    modifies = "weights"
    probeable = ("error", "activities", "delta", "pos_memristors", "neg_memristors")
    
    pre_synapse = SynapseParam( "pre_synapse", default=Lowpass( tau=0.005 ), readonly=True )
    post_synapse = SynapseParam( "post_synapse", default=None, readonly=True )
    beta = NumberParam( "beta", low=0, readonly=True, default=1.0 )
    r_max = NumberParam( "r_max", readonly=True, default=2.3e8 )
    r_min = NumberParam( "r_min", readonly=True, default=200 )
    exponent = NumberParam( "exponent", readonly=True, default=-0.146 )
    gain = NumberParam( "gain", readonly=True, default=1e3 )
    
    def __init__( self,
                  pre_synapse=Default,
                  post_synapse=Default,
                  beta=Default,
                  r_max=Default,
                  r_min=Default,
                  exponent=Default,
                  noisy=False,
                  gain=Default,
                  seed=None ):
        super().__init__( size_in="post_state" )
        
        self.pre_synapse = pre_synapse
        self.post_synapse = post_synapse
        self.beta = beta
        self.r_max = r_max
        self.r_min = r_min
        self.exponent = exponent
        if not noisy:
            self.noise_percentage = np.zeros( 4 )
        elif isinstance( noisy, float ):
            self.noise_percentage = np.full( 4, noisy )
        elif isinstance( noisy, list ) and len( noisy ) == 4:
            self.noise_percentage = noisy
        else:
            raise ValueError( f"Noisy parameter must be int or list of length 4, not {type( noisy )}" )
        self.gain = gain
        self.seed = seed
    
    @property
    def _argdefaults( self ):
        return (
                ("pre_synapse", mOja.pre_synapse.default),
                ("post_synapse", mOja.post_synapse.default),
                ("beta", mOja.beta.default),
                ("r_max", mOja.r_max.default),
                ("r_min", mOja.r_min.default),
                ("exponent", mOja.exponent.default),
                )


class SimmOja( Operator ):
    def __init__(
            self,
            pre_filtered,
            post_filtered,
            beta,
            pos_memristors,
            neg_memristors,
            weights,
            noise_percentage,
            gain,
            r_min,
            r_max,
            exponent,
            states=None,
            tag=None
            ):
        super( SimmOja, self ).__init__( tag=tag )
        
        # scale beta by gain for it to have the expected effect
        self.beta = beta * gain / 2
        self.noise_percentage = noise_percentage
        self.gain = gain
        self.r_min = r_min
        self.r_max = r_max
        self.exponent = exponent
        
        self.sets = [ ] + ([ ] if states is None else [ states ])
        self.incs = [ ]
        self.reads = [ pre_filtered, post_filtered ]
        self.updates = [ weights, pos_memristors, neg_memristors ]
    
    @property
    def pre_filtered( self ):
        return self.reads[ 0 ]
    
    @property
    def post_filtered( self ):
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
        return "pre=%s, post=%s -> %s" % (self.pre_filtered, self.post_filtered, self.weights)
    
    def make_step( self, signals, dt, rng ):
        pre_filtered = signals[ self.pre_filtered ]
        post_filtered = signals[ self.post_filtered ]
        
        pos_memristors = signals[ self.pos_memristors ]
        neg_memristors = signals[ self.neg_memristors ]
        weights = signals[ self.weights ]
        
        beta = self.beta
        gain = self.gain
        r_min = self.r_min
        r_max = self.r_max
        exponent = self.exponent
        
        # overwrite initial transform with memristor-based weights
        weights[ : ] = gain * \
                       (resistance2conductance( pos_memristors, r_min, r_max )
                        - resistance2conductance( neg_memristors, r_min, r_max ))
        
        def step_simmoja():
            # calculate the magnitude of the update based on PES learning rule
            # local_error = -np.dot( encoders, error )
            # I can use NengoDL build function like this, as dot(encoders, error) has been done there already
            # i.e., error already contains the PES local error
            post_squared = post_filtered * post_filtered
            forgetting = beta * weights * post_squared
            hebbian = np.outer( post_filtered, pre_filtered )
            oja_delta = hebbian - forgetting
            
            # some memristors are adjusted erroneously if we don't filter
            spiked_map = find_spikes( pre_filtered, weights.shape, post_filtered, invert=True )
            oja_delta[ spiked_map ] = 0
            
            # set update direction and magnitude (unused with powerlaw memristor equations)
            V = np.sign( oja_delta ) * 1e-1
            
            # clip values outside [R_0,R_1]
            clip_memristor_values( V, pos_memristors, neg_memristors, r_max, r_min )
            
            # update the two memristor pairs
            update_memristors( V, pos_memristors, neg_memristors, r_max, r_min, exponent )
            
            # update network weights
            update_weights( V, weights, pos_memristors, neg_memristors, r_max, r_min, gain )
        
        return step_simmoja


class mPES( LearningRuleType ):
    modifies = "weights"
    probeable = ("error", "activities", "delta", "pos_memristors", "neg_memristors")
    
    pre_synapse = SynapseParam( "pre_synapse", default=Lowpass( tau=0.005 ), readonly=True )
    r_max = NumberParam( "r_max", readonly=True, default=2.3e8 )
    r_min = NumberParam( "r_min", readonly=True, default=200 )
    exponent = NumberParam( "exponent", readonly=True, default=-0.146 )
    gain = NumberParam( "gain", readonly=True, default=1e3 )
    
    def __init__( self,
                  pre_synapse=Default,
                  r_max=Default,
                  r_min=Default,
                  exponent=Default,
                  noisy=False,
                  gain=Default,
                  seed=None ):
        super().__init__( size_in="post_state" )
        
        self.pre_synapse = pre_synapse
        self.r_max = r_max
        self.r_min = r_min
        self.exponent = exponent
        if not noisy:
            self.noise_percentage = np.zeros( 4 )
        elif isinstance( noisy, float ):
            self.noise_percentage = np.full( 4, noisy )
        elif isinstance( noisy, list ) and len( noisy ) == 4:
            self.noise_percentage = noisy
        else:
            raise ValueError( f"Noisy parameter must be int or list of length 4, not {type( noisy )}" )
        self.gain = gain
        self.seed = seed
    
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
            pos_memristors,
            neg_memristors,
            weights,
            noise_percentage,
            gain,
            r_min,
            r_max,
            exponent,
            states=None,
            tag=None
            ):
        super( SimmPES, self ).__init__( tag=tag )
        
        self.noise_percentage = noise_percentage
        self.gain = gain
        self.error_threshold = 1e-5
        self.r_min = r_min
        self.r_max = r_max
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
        error_threshold = self.error_threshold
        r_min = self.r_min
        r_max = self.r_max
        exponent = self.exponent
        
        # overwrite initial transform with memristor-based weights
        weights[ : ] = gain * \
                       (resistance2conductance( pos_memristors, r_min, r_max )
                        - resistance2conductance( neg_memristors, r_min, r_max ))
        
        def step_simmpes():
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
                
                # clip values outside [R_0,R_1]
                clip_memristor_values( V, pos_memristors, neg_memristors, r_max, r_min )
                
                # update the two memristor pairs
                update_memristors( V, pos_memristors, neg_memristors, r_max, r_min, exponent )
                
                # update network weights
                update_weights( V, weights, pos_memristors, neg_memristors, r_max, r_min, gain )
        
        return step_simmpes


################ NENGO DL #####################

import tensorflow as tf
from nengo.builder import Signal
from nengo.builder.operator import Reset, DotInc, Copy

from nengo_dl.builder import Builder, OpBuilder, NengoBuilder
from nengo.builder import Builder as NengoCoreBuilder


@NengoCoreBuilder.register( mOja )
def build_moja( model, moja, rule ):
    conn = rule.connection
    pre_activities = model.sig[ get_pre_ens( conn ).neurons ][ "out" ]
    post_activities = model.sig[ get_post_ens( conn ).neurons ][ "out" ]
    pre_filtered = build_or_passthrough( model, moja.pre_synapse, pre_activities )
    post_filtered = build_or_passthrough( model, moja.post_synapse, post_activities )
    
    pos_memristors, neg_memristors, r_min_noisy, r_max_noisy, exponent_noisy = initialise_memristors( moja,
                                                                                                      pre_filtered.shape[
                                                                                                          0 ],
                                                                                                      post_filtered.shape[
                                                                                                          0 ] )
    
    model.sig[ conn ][ "pos_memristors" ] = pos_memristors
    model.sig[ conn ][ "neg_memristors" ] = neg_memristors
    
    model.add_op(
            SimmOja(
                    pre_filtered,
                    post_filtered,
                    moja.beta,
                    model.sig[ conn ][ "pos_memristors" ],
                    model.sig[ conn ][ "neg_memristors" ],
                    model.sig[ conn ][ "weights" ],
                    moja.noise_percentage,
                    moja.gain,
                    r_min_noisy,
                    r_max_noisy,
                    exponent_noisy
                    )
            )
    
    # expose these for probes
    model.sig[ rule ][ "pre_filtered" ] = pre_filtered
    model.sig[ rule ][ "post_filtered" ] = post_filtered
    model.sig[ rule ][ "pos_memristors" ] = pos_memristors
    model.sig[ rule ][ "neg_memristors" ] = neg_memristors


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
    
    pos_memristors, neg_memristors, r_min_noisy, r_max_noisy, exponent_noisy = initialise_memristors( mpes,
                                                                                                      acts.shape[ 0 ],
                                                                                                      encoders.shape[
                                                                                                          0 ] )
    
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
                     model.sig[ conn ][ "pos_memristors" ],
                     model.sig[ conn ][ "neg_memristors" ],
                     model.sig[ conn ][ "weights" ],
                     mpes.noise_percentage,
                     mpes.gain,
                     r_min_noisy,
                     r_max_noisy,
                     exponent_noisy )
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
        self.r_min = signals.op_constant( self.ops,
                                          [ 1 for _ in self.ops ],
                                          "r_min",
                                          signals.dtype,
                                          shape=(1, -1, 1, 1) )
        self.r_min = tf.reshape( self.r_min,
                                 (1,
                                  len( self.ops ),
                                  self.ops[ 0 ].r_min.shape[ 0 ],
                                  self.ops[ 0 ].r_min.shape[ 1 ])
                                 )
        self.r_max = signals.op_constant( self.ops,
                                          [ 1 for _ in self.ops ],
                                          "r_max",
                                          signals.dtype,
                                          shape=(1, -1, 1, 1) )
        self.r_max = tf.reshape( self.r_max,
                                 (1,
                                  len( self.ops ),
                                  self.ops[ 0 ].r_max.shape[ 0 ],
                                  self.ops[ 0 ].r_max.shape[ 1 ])
                                 )
        self.exponent = signals.op_constant( self.ops,
                                             [ 1 for _ in self.ops ],
                                             "exponent",
                                             signals.dtype,
                                             shape=(1, -1, 1, 1) )
        self.exponent = tf.reshape( self.exponent,
                                    (1,
                                     len( self.ops ),
                                     self.ops[ 0 ].exponent.shape[ 0 ],
                                     self.ops[ 0 ].exponent.shape[ 1 ])
                                    )
        self.error_threshold = signals.op_constant( self.ops,
                                                    [ 1 for _ in self.ops ],
                                                    "error_threshold",
                                                    signals.dtype,
                                                    shape=(1, -1, 1, 1) )
        self.g_min = 1.0 / self.r_max
        self.g_max = 1.0 / self.r_min
    
    def build_step( self, signals ):
        pre_filtered = signals.gather( self.pre_data )
        local_error = signals.gather( self.error_data )
        pos_memristors = signals.gather( self.pos_memristors )
        neg_memristors = signals.gather( self.neg_memristors )
        
        r_min = self.r_min
        r_max = self.r_max
        exponent = self.exponent
        
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
        
        # @tf.function
        def update_resistances( pos_memristors, neg_memristors ):
            pos_mask = tf.greater( V, 0 )
            pos_indices = tf.where( pos_mask )
            neg_mask = tf.less( V, 0 )
            neg_indices = tf.where( neg_mask )
            
            # clip values outside [R_0,R_1]
            pos_memristors = tf.tensor_scatter_nd_update( pos_memristors,
                                                          pos_indices,
                                                          tf.where(
                                                                  tf.greater(
                                                                          tf.boolean_mask( pos_memristors, pos_mask ),
                                                                          tf.boolean_mask( r_max, pos_mask ) ),
                                                                  tf.boolean_mask( r_max, pos_mask ),
                                                                  tf.boolean_mask( pos_memristors, pos_mask ) ) )
            pos_memristors = tf.tensor_scatter_nd_update( pos_memristors,
                                                          pos_indices,
                                                          tf.where(
                                                                  tf.less( tf.boolean_mask( pos_memristors, pos_mask ),
                                                                           tf.boolean_mask( r_min, pos_mask ) ),
                                                                  tf.boolean_mask( r_min, pos_mask ),
                                                                  tf.boolean_mask( pos_memristors, pos_mask ) ) )
            neg_memristors = tf.tensor_scatter_nd_update( neg_memristors,
                                                          neg_indices,
                                                          tf.where(
                                                                  tf.greater(
                                                                          tf.boolean_mask( neg_memristors, neg_mask ),
                                                                          tf.boolean_mask( r_max, neg_mask ) ),
                                                                  tf.boolean_mask( r_max, neg_mask ),
                                                                  tf.boolean_mask( neg_memristors, neg_mask ) ) )
            neg_memristors = tf.tensor_scatter_nd_update( neg_memristors,
                                                          neg_indices,
                                                          tf.where(
                                                                  tf.less( tf.boolean_mask( neg_memristors, neg_mask ),
                                                                           tf.boolean_mask( r_min, neg_mask ) ),
                                                                  tf.boolean_mask( r_min, neg_mask ),
                                                                  tf.boolean_mask( neg_memristors, neg_mask ) ) )
            
            # positive memristors update
            pos_n = tf.math.pow( (tf.boolean_mask( pos_memristors, pos_mask ) - tf.boolean_mask( r_min, pos_mask ))
                                 / tf.boolean_mask( r_max, pos_mask ),
                                 1 / tf.boolean_mask( exponent, pos_mask ) )
            pos_update = tf.boolean_mask( r_min, pos_mask ) + tf.boolean_mask( r_max, pos_mask ) * \
                         tf.math.pow( pos_n + 1, tf.boolean_mask( exponent, pos_mask ) )
            pos_memristors = tf.tensor_scatter_nd_update( pos_memristors, pos_indices, pos_update )
            
            # negative memristors update
            neg_n = tf.math.pow( (tf.boolean_mask( neg_memristors, neg_mask ) - tf.boolean_mask( r_min, neg_mask ))
                                 / tf.boolean_mask( r_max, neg_mask ),
                                 1 / tf.boolean_mask( exponent, neg_mask ) )
            neg_update = tf.boolean_mask( r_min, neg_mask ) + tf.boolean_mask( r_max, neg_mask ) * \
                         tf.math.pow( neg_n + 1, tf.boolean_mask( exponent, neg_mask ) )
            neg_memristors = tf.tensor_scatter_nd_update( neg_memristors, neg_indices, neg_update )
            
            return pos_memristors, neg_memristors
        
        pes_delta = -local_error * pre_filtered
        
        spiked_map = find_spikes( pre_filtered, self.output_size )
        pes_delta = pes_delta * spiked_map
        
        V = tf.sign( pes_delta ) * 1e-1
        
        # if any errors is above threshold then pass update resistances
        # if all errors are below threshold then do nothing
        pos_memristors, neg_memristors = tf.cond(
                tf.reduce_any( tf.greater( tf.abs( local_error ), self.error_threshold ) ),
                true_fn=lambda: update_resistances( pos_memristors,
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
                and x.local_error.shape[ 0 ] == y.local_error.shape[ 0 ]
        )
