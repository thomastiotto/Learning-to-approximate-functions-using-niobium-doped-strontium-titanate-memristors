import warnings

from nengo.params import (
    Default,
    NumberParam,
    )
from nengo.synapses import Lowpass, SynapseParam
from nengo.learning_rules import LearningRuleType

import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import Reset
from nengo.ensemble import Ensemble
from nengo.learning_rules import PES
from nengo.node import Node


class mPES( LearningRuleType ):
    modifies = "weights"
    probeable = ("error", "activities", "delta")
    
    learning_rate = NumberParam( "learning_rate", low=0, readonly=True, default=1e-4 )
    pre_synapse = SynapseParam( "pre_synapse", default=Lowpass( tau=0.005 ), readonly=True )
    
    def __init__( self, memristor_model, learning_rate=Default, pre_synapse=Default ):
        super().__init__( learning_rate, size_in="post_state" )
        if learning_rate is not Default and learning_rate >= 1.0:
            warnings.warn(
                    "This learning rate is very high, and can result "
                    "in floating point errors from too much current."
                    )
        
        self.pre_synapse = pre_synapse
        self.memristor_model = memristor_model
    
    @property
    def _argdefaults( self ):
        return (
                ("learning_rate", PES.learning_rate.default),
                ("pre_synapse", PES.pre_synapse.default),
                )


class SimmPES( Operator ):
    def __init__(
            self, pre_filtered, error, delta, learning_rate, encoders, memristor_model, tag=None
            ):
        super( SimmPES, self ).__init__( tag=tag )
        
        self.learning_rate = learning_rate
        self.memristor_model = memristor_model
        
        # create memristor array that implement the weights
        self.memristors = np.empty_like( delta.initial_value, dtype=object )
        for i in range( self.memristors.shape[ 0 ] ):
            for j in range( self.memristors.shape[ 1 ] ):
                self.memristors[ i, j ] = self.memristor_model()
        
        self.sets = [ ]
        self.incs = [ ]
        self.reads = [ pre_filtered, error, encoders ]
        self.updates = [ delta ]
    
    @property
    def delta( self ):
        return self.updates[ 0 ]
    
    @property
    def encoders( self ):
        return None if len( self.reads ) < 3 else self.reads[ 2 ]
    
    @property
    def error( self ):
        return self.reads[ 1 ]
    
    @property
    def pre_filtered( self ):
        return self.reads[ 0 ]
    
    def _descstr( self ):
        return "pre=%s, error=%s -> %s" % (self.pre_filtered, self.error, self.delta)
    
    def make_step( self, signals, dt, rng ):
        pre_filtered = signals[ self.pre_filtered ]
        error = signals[ self.error ]
        delta = signals[ self.delta ]
        n_neurons = pre_filtered.shape[ 0 ]
        alpha = -self.learning_rate * dt / n_neurons
        encoders = signals[ self.encoders ]
        memristors = self.memristors
        
        def find_spikes( input_activities, output_activities=None ):
            spiked_pre = np.tile(
                    np.array( np.rint( input_activities ), dtype=bool ), (input_activities.shape[ 0 ], 1)
                    )
            spiked_post = np.tile(
                    np.expand_dims(
                            np.array( np.rint( output_activities ), dtype=bool ), axis=1 ),
                    (1, input_activities.shape[ 0 ])
                    ) if output_activities is not None else np.ones( (1, input_activities.shape[ 0 ]) )
            
            return np.logical_and( spiked_pre, spiked_post )
        
        def step_simmpes():
            local_error = alpha * np.dot( encoders, error )
            signal = np.outer( local_error, pre_filtered )
            
            spiked_map = find_spikes( pre_filtered )
            
            if spiked_map.any():
                for j, i in np.transpose( np.where( spiked_map ) ):
                    update = self.memristors[ j, i ].pulse( signal[ j, i ], delta=True )
                    # update = update if update >=
                    delta[ j, i ] = update
            
            # np.outer( alpha * np.dot( encoders, error ), pre_filtered, out=delta )
        
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
def build_mpes( model, pes, rule ):
    conn = rule.connection
    
    # NB "pes" is the "mPES()" frontend class
    
    # Create input error signal
    error = Signal( shape=rule.size_in, name="PES:error" )
    model.add_op( Reset( error ) )
    model.sig[ rule ][ "in" ] = error  # error connection will attach here
    
    # Filter pre-synaptic activities with pre_synapse
    acts = build_or_passthrough( model, pes.pre_synapse, model.sig[ conn.pre_obj ][ "out" ] )
    
    post = get_post_ens( conn )
    encoders = model.sig[ post ][ "encoders" ][ :, conn.post_slice ]
    
    model.add_op(
            SimmPES(
                    acts, error, model.sig[ rule ][ "delta" ], pes.learning_rate, encoders=encoders,
                    memristor_model=pes.memristor_model
                    )
            )
    
    # expose these for probes
    model.sig[ rule ][ "error" ] = error
    model.sig[ rule ][ "activities" ] = acts
