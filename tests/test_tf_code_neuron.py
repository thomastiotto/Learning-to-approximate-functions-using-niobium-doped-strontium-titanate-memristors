import tensorflow as tf
import numpy as np
import copy
from nengo.utils.numpy import clip

tau_rc = 0.02
min_voltage = 0
amplitude = 1
tau_inhibition = 10
tau_ref = 0.002
tau_n = 1
inc_n = 0.01
dt = 0.001


def np_calc( dt, J, output, voltage, refractory_time, adaptation, inhibition ):
    """Implement the AdaptiveLIF nonlinearity."""
    
    J = J - adaptation
    
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
    output[ : ] = spiked_mask * (amplitude / dt)
    
    # if neuron that spiked had highest input but was still inhibited from a previous timestep
    voltage[ inhibition != 0 ] = 0
    output[ inhibition != 0 ] = 0
    spiked_mask[ inhibition != 0 ] = False
    
    if np.count_nonzero( output ) > 0:
        # inhibit all other neurons than one with highest input
        voltage[ J != np.max( J ) ] = 0
        output[ J != np.max( J ) ] = 0
        spiked_mask[ J != np.max( J ) ] = False
        inhibition[ (J != np.max( J )) & (inhibition == 0) ] = tau_inhibition
    
    # set v(0) = 1 and solve for t to compute the spike time
    t_spike = dt + tau_rc * np.log1p(
            -(voltage[ spiked_mask ] - 1) / (J[ spiked_mask ] - 1)
            )
    
    # set spiked voltages to zero, refractory times to tau_ref, and
    # rectify negative voltages to a floor of min_voltage
    voltage[ voltage < min_voltage ] = min_voltage
    voltage[ spiked_mask ] = 0
    refractory_time[ spiked_mask ] = tau_ref + t_spike
    
    adaptation += (dt / tau_n) * (inc_n * output - adaptation)
    
    inhibition[ inhibition != 0 ] -= 1
    
    return J, output, voltage, refractory_time, adaptation, inhibition


def tf_calc( dt, J, voltage, refractory_time, adaptation, inhibition ):
    zero = tf.constant( 0., dtype=tf.float64 )
    one = tf.constant( 1., dtype=tf.float64 )
    alpha = tf.constant( amplitude / dt, dtype=tf.float64 )
    zeros = tf.zeros_like( J_tf_in, dtype=tf.float64 )
    tau_ref = tf.constant( 0.002, dtype=tf.float64 )
    
    """Implement the AdaptiveLIF nonlinearity."""
    
    def inhibit( voltage, output, inhibition, spiked_mask ):
        J_mask = tf.equal( J, tf.reduce_max( J ) )
        
        voltage = tf.multiply( voltage, tf.cast( J_mask, voltage.dtype ) )
        output = tf.multiply( output, tf.cast( J_mask, output.dtype ) )
        spiked_mask = tf.logical_and( spiked_mask, tf.cast( J_mask, spiked_mask.dtype ) )
        inhibition = tf.where( tf.logical_and( tf.logical_not( J_mask ), tf.equal( inhibition, 0 ) ),
                               tau_inhibition,
                               inhibition
                               )
        
        return voltage, output, inhibition, spiked_mask
    
    J = J - adaptation
    
    # compute effective dt for each neuron, based on remaining time.
    # note that refractory times that have completed midway into this
    # timestep will be given a partial timestep, and moreover these will
    # be subtracted to zero at the next timestep (or reset by a spike)
    delta_t = tf.clip_by_value( dt - refractory_time, zero, dt )
    
    # update voltage using discretized lowpass filter
    # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
    # J is constant over the interval [t, t + dt)
    dV = (voltage - J) * tf.math.expm1(
            -delta_t / tau_rc  # pylint: disable=invalid-unary-operand-type
            )
    voltage += dV
    
    # determine which neurons spiked (set them to 1/dt, else 0)
    spiked_mask = voltage > one
    output = tf.cast( spiked_mask, J.dtype ) * alpha
    
    inhibition_mask = tf.equal( inhibition, 0 )
    # if neuron that spiked had highest input but was still inhibited from a previous timestep
    voltage = tf.multiply( voltage, tf.cast( inhibition_mask, voltage.dtype ) )
    output = tf.multiply( output, tf.cast( inhibition_mask, output.dtype ) )
    spiked_mask = tf.logical_and( spiked_mask, tf.cast( inhibition_mask, spiked_mask.dtype ) )
    
    # inhibit all other neurons than one with highest input
    voltage, output, inhibition, spiked_mask = tf.cond( tf.math.count_nonzero( output ) > 0,
                                                        lambda: inhibit( voltage, output, inhibition, spiked_mask ),
                                                        lambda: (tf.identity( voltage ),
                                                                 tf.identity( output ),
                                                                 tf.identity( inhibition ),
                                                                 tf.identity( spiked_mask )
                                                                 )
                                                        )
    
    # set v(0) = 1 and solve for t to compute the spike time
    t_spike = dt + tau_rc * tf.math.log1p( -(voltage - 1) / (J - 1) )
    
    # set spiked voltages to zero, refractory times to tau_ref, and
    # rectify negative voltages to a floor of min_voltage
    voltage = tf.where( spiked_mask, zeros, tf.maximum( voltage, min_voltage ) )
    refractory_time = tf.where( spiked_mask, tau_ref + t_spike, refractory_time - dt )
    
    adaptation += (dt / tau_n) * (inc_n * output - adaptation)
    
    inhibition_mask = tf.not_equal( inhibition, 0 )
    inhibition = tf.tensor_scatter_nd_sub( inhibition,
                                           tf.where( inhibition_mask ),
                                           tf.ones( tf.math.count_nonzero( inhibition_mask ), dtype=tf.float64 )
                                           )
    
    return J, output, voltage, refractory_time, adaptation, inhibition


J_np_in = np.array( [ 221.90912353, 185.73059046, 187.77355607 ] )
adaptation_np_in = np.array( [ 0., 0., 0.00996006 ] )
inhibition_np_in = np.array( [ 0., 0., 7. ] )
output_np_in = np.array( [ 0., 0., 0. ] )
refractory_time_np_in = np.array( [ -0.007, -0.007, -0.00173687 ] )
voltage_np_in = np.array( [ 0.51768583, 0.24231603, 0.19985032 ] )

J_tf_in = tf.reshape( tf.convert_to_tensor( J_np_in ),
                      [ 1, J_np_in.shape[ 0 ] ] )
adaptation_tf_in = tf.reshape( tf.convert_to_tensor( adaptation_np_in ),
                               [ 1, J_np_in.shape[ 0 ] ] )
inhibition_tf_in = tf.reshape( tf.convert_to_tensor( inhibition_np_in ),
                               [ 1, J_np_in.shape[ 0 ] ] )
refractory_time_tf_in = tf.reshape( tf.convert_to_tensor( refractory_time_np_in ),
                                    [ 1, J_np_in.shape[ 0 ] ] )
voltage_tf_in = tf.reshape( tf.convert_to_tensor( voltage_np_in ),
                            [ 1, J_np_in.shape[ 0 ] ] )

# tf.compat.v1.disable_eager_execution()

J_np_out, output_np_out, voltage_np_out, refractory_time_np_out, adaptation_np_out, inhibition_np_out = np_calc( dt,
                                                                                                                 copy.deepcopy(
                                                                                                                         J_np_in ),
                                                                                                                 copy.deepcopy(
                                                                                                                         output_np_in ),
                                                                                                                 copy.deepcopy(
                                                                                                                         voltage_np_in ),
                                                                                                                 copy.deepcopy(
                                                                                                                         refractory_time_np_in ),
                                                                                                                 copy.deepcopy(
                                                                                                                         adaptation_np_in ),
                                                                                                                 copy.deepcopy(
                                                                                                                         inhibition_np_in ) )
J_tf_out, output_tf_out, voltage_tf_out, refractory_time_tf_out, adaptation_tf_out, inhibition_tf_out = tf_calc( dt,
                                                                                                                 copy.deepcopy(
                                                                                                                         J_tf_in ),
                                                                                                                 copy.deepcopy(
                                                                                                                         voltage_tf_in ),
                                                                                                                 copy.deepcopy(
                                                                                                                         refractory_time_tf_in ),
                                                                                                                 copy.deepcopy(
                                                                                                                         adaptation_tf_in ),
                                                                                                                 copy.deepcopy(
                                                                                                                         inhibition_tf_in ) )

print( "NumPy\n", J_np_out, "\n", output_np_out, "\n", voltage_np_out, "\n", refractory_time_np_out, "\n",
       adaptation_np_out, "\n", inhibition_np_out )
print( "TensorFlow\n", J_tf_out.numpy().squeeze(), "\n", output_tf_out.numpy().squeeze(), "\n",
       voltage_tf_out.numpy().squeeze(), "\n", refractory_time_tf_out.numpy().squeeze(), "\n",
       adaptation_tf_out.numpy().squeeze(), "\n", inhibition_tf_out.numpy().squeeze() )
print( "tf and np are equal?",
       np.array_equal( J_np_out, J_tf_out.numpy().squeeze() )
       and np.array_equal( voltage_np_out, voltage_tf_out.numpy().squeeze() )
       and np.array_equal( refractory_time_np_out, refractory_time_tf_out.numpy().squeeze() )
       and np.array_equal( adaptation_np_out, adaptation_tf_out.numpy().squeeze() )
       and np.array_equal( inhibition_np_out, inhibition_tf_out.numpy().squeeze() )
       )
