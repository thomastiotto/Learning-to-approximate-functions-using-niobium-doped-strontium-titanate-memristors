import numpy as np

from nengo.neurons import LIF
from nengo.params import NumberParam
from nengo.dists import Uniform, Choice
from nengo.builder import Operator

# numpy 1.17 introduced a slowdown to clip, so
# use nengo.utils.numpy.clip instead of np.clip
# This has persisted through 1.19 at least
clip = (
        np.core.umath.clip
        if tuple( int( st ) for st in np.__version__.split( "." ) ) >= (1, 17, 0)
        else np.clip
)


class AdaptiveLIFLateralInhibition( LIF ):
    state = {
            "voltage"        : Uniform( low=0, high=1 ),
            "refractory_time": Choice( [ 0 ] ),
            "adaptation"     : Choice( [ 0 ] ),
            "inhibition"     : Choice( [ 0 ] )
            }
    spiking = True
    
    tau_n = NumberParam( "tau_n", low=0, low_open=True )
    inc_n = NumberParam( "inc_n", low=0 )
    
    def __init__(
            self,
            tau_n=1,
            inc_n=0.01,
            tau_rc=0.02,
            tau_ref=0.002,
            min_voltage=0,
            amplitude=1,
            initial_state=None,
            inhibition=10
            ):
        super().__init__(
                tau_rc=tau_rc,
                tau_ref=tau_ref,
                min_voltage=min_voltage,
                amplitude=amplitude,
                initial_state=initial_state,
                )
        self.tau_n = tau_n
        self.inc_n = inc_n
        self.inhibition = inhibition
    
    def step( self, dt, J, output, voltage, refractory_time, adaptation, inhibition ):
        """Implement the AdaptiveLIF nonlinearity."""
        
        n = adaptation
        J = J - n
        
        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage
        
        # reduce all refractory times by dt
        refractory_time -= dt
        
        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip( (dt - refractory_time), 0, dt )
        
        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1( -delta_t / tau_rc )
        
        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        output[ : ] = spiked_mask * (self.amplitude / dt)
        
        # if neuron that spiked had highest input but was still inhibited from a previous timestep
        voltage[ inhibition != 0 ] = 0
        output[ inhibition != 0 ] = 0
        spiked_mask[ inhibition != 0 ] = False
        
        if np.count_nonzero( output ) > 0:
            # inhibit all other neurons than one with highest input
            voltage[ J != np.max( J ) ] = 0
            output[ J != np.max( J ) ] = 0
            spiked_mask[ J != np.max( J ) ] = False
            inhibition[ (J != np.max( J )) & (inhibition == 0) ] = self.inhibition
        
        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
                -(voltage[ spiked_mask ] - 1) / (J[ spiked_mask ] - 1)
                )
        
        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[ voltage < min_voltage ] = min_voltage
        voltage[ spiked_mask ] = 0
        refractory_time[ spiked_mask ] = self.tau_ref + t_spike
        
        n += (dt / self.tau_n) * (self.inc_n * output - n)
        
        inhibition[ inhibition != 0 ] -= 1


import tensorflow as tf
from nengo_dl.neuron_builders import SoftLIFRateBuilder, LIFRateBuilder


class AdaptiveLIFLateralInhibitionBuilder( SoftLIFRateBuilder ):
    """Build a group of `~nengo.LIF` neuron operators."""
    
    spiking = True
    
    def build_pre( self, signals, config ):
        # note: we skip the SoftLIFRateBuilder init
        # pylint: disable=bad-super-call
        super( SoftLIFRateBuilder, self ).build_pre( signals, config )
        
        self.min_voltage = signals.op_constant(
                [ op.neurons for op in self.ops ],
                [ op.J.shape[ 0 ] for op in self.ops ],
                "min_voltage",
                signals.dtype,
                )
        print( "diocane" )
        
        if self.config.lif_smoothing:
            self.sigma = tf.constant( self.config.lif_smoothing, dtype=signals.dtype )
    
    def step( self, J, dt, voltage, refractory_time ):
        delta_t = tf.clip_by_value( dt - refractory_time, self.zero, dt )
        
        dV = (voltage - J) * tf.math.expm1(
                -delta_t / self.tau_rc  # pylint: disable=invalid-unary-operand-type
                )
        voltage += dV
        
        spiked = voltage > self.one
        spikes = tf.cast( spiked, J.dtype ) * self.alpha
        
        partial_ref = -self.tau_rc * tf.math.log1p(
                (self.one - voltage) / (J - self.one)
                )
        # FastLIF version (linearly approximate spike time when calculating
        # remaining refractory period)
        # partial_ref = signals.dt * (voltage - self.one) / dV
        
        refractory_time = tf.where(
                spiked, self.tau_ref - partial_ref, refractory_time - dt
                )
        
        voltage = tf.where( spiked, self.zeros, tf.maximum( voltage, self.min_voltage ) )
        
        return spikes, voltage, refractory_time
    
    def training_step( self, J, dt, **state ):
        return (
                LIFRateBuilder.step( self, J, dt )
                if self.config.lif_smoothing is None
                else SoftLIFRateBuilder.step( self, J, dt )
        )


# register with the NengoDL neuron builder
from nengo_dl.neuron_builders import SimNeuronsBuilder

SimNeuronsBuilder.TF_NEURON_IMPL[ AdaptiveLIFLateralInhibition ] = AdaptiveLIFLateralInhibitionBuilder
