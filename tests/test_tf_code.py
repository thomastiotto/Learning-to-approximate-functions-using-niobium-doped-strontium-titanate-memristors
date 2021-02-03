import tensorflow as tf
import numpy as np
import copy


def np_calc( local_error, pre_filtered, pos_memristors, neg_memristors, r_min, r_max, exponent ):
    g_min = 1.0 / r_max
    g_max = 1.0 / r_min
    error_threshold = 1e-5
    gain = 1e4
    np.random.seed( 0 )
    
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
    
    if np.any( np.absolute( local_error ) > error_threshold ):
        # calculate the magnitude of the update based on PES learning rule
        # local_error = -np.dot( encoders, error )
        # I can use NengoDL build function like this, as dot(encoders, error) has been done there already
        # i.e., error already contains the PES local error
        pes_delta = np.outer( -local_error, pre_filtered )
        
        # some memristors are adjusted erroneously if we don't filter
        spiked_map = find_spikes( pre_filtered, (output_size, input_size), invert=True )
        pes_delta[ spiked_map ] = 0
        
        # set update direction and magnitude (unused with powerlaw memristor equations)
        V = np.sign( pes_delta ) * 1e-1
        
        # clip values outside [R_0,R_1]
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
        
        # update the two memristor pairs separately
        pos_n = np.power( (pos_memristors[ V > 0 ] - r_min[ V > 0 ]) / r_max[ V > 0 ],
                          1 / exponent[ V > 0 ] )
        pos_memristors[ V > 0 ] = r_min[ V > 0 ] + r_max[ V > 0 ] * np.power( pos_n + 1, exponent[ V > 0 ] )
        
        neg_n = np.power( (neg_memristors[ V < 0 ] - r_min[ V < 0 ]) / r_max[ V < 0 ], 1 / exponent[ V < 0 ] )
        neg_memristors[ V < 0 ] = r_min[ V < 0 ] + r_max[ V < 0 ] * np.power( neg_n + 1, exponent[ V < 0 ] )
    
    return \
        resistance2conductance( pos_memristors[ : ] ) - resistance2conductance( neg_memristors[ : ] ), \
        pos_memristors, \
        neg_memristors


def tf_calc( local_error, pre_filtered, pos_memristors, neg_memristors, r_min, r_max, exponent ):
    g_min = 1.0 / r_max
    g_max = 1.0 / r_min
    error_threshold = 1e-5
    gain = 1e4
    
    tf.random.set_seed( 0 )
    
    def resistance2conductance( R ):
        g_curr = 1.0 / R
        g_norm = (g_curr - g_min) / (g_max - g_min)
        
        return g_norm * gain
    
    def find_spikes( input_activities, output_size, invert=False ):
        spiked_pre = tf.cast(
                tf.tile( tf.math.rint( input_activities ), [ 1, 1, output_size, 1 ] ),
                tf.bool )
        
        out = spiked_pre
        if invert:
            out = tf.math.logical_not( out )
        
        return tf.cast( out, tf.float64 )
    
    def update_resistances( pos_memristors, neg_memristors ):
        pos_mask = tf.greater( V, 0 )
        pos_indices = tf.where( pos_mask )
        neg_mask = tf.less( V, 0 )
        neg_indices = tf.where( neg_mask )
        
        # clip values outside [R_0,R_1]
        pos_memristors = tf.tensor_scatter_nd_update( pos_memristors,
                                                      pos_indices,
                                                      tf.where(
                                                              tf.greater( tf.boolean_mask( pos_memristors, pos_mask ),
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
                                                              tf.greater( tf.boolean_mask( neg_memristors, neg_mask ),
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
    
    spiked_map = find_spikes( pre_filtered, output_size )
    pes_delta = pes_delta * spiked_map
    
    V = tf.sign( pes_delta ) * 1e-1
    
    # check if the error is greater than the threshold
    # if it is not then pass decision to next tf.cond()
    # if it is then do nothing
    pos_memristors, neg_memristors = tf.cond( tf.reduce_any( tf.greater( tf.abs( local_error ), error_threshold ) ),
                                              true_fn=lambda: update_resistances( pos_memristors, neg_memristors ),
                                              false_fn=lambda: (
                                                      tf.identity( pos_memristors ),
                                                      tf.identity( neg_memristors ))
                                              )
    
    return \
        resistance2conductance( pos_memristors ) - resistance2conductance( neg_memristors ), \
        pos_memristors, \
        neg_memristors


# tf.compat.v1.disable_eager_execution()

output_size = input_size = 2

local_error_np = np.array(
        # [ -8.60119821, 10.69191986, 19.24545064, -2.81894391 ]
        [ -4.53326035, 8.77298927 ]
        )
pre_filtered_np = np.array(
        # [ 0., 0., 0., 148.41070704 ]
        [ 215.095932, 0 ]
        )
pos_memristors_np = np.array(
        [ [ 262434544, 38824360 ],
          [ 71839.4141, -7296862 ] ]
        # [ [ 1.05928446e+08, 1.08442657e+08, 1.08579456e+08, 1.08472517e+08 ],
        #   [ 1.06235637e+08, 1.03843817e+08, 1.02975346e+08, 1.00567130e+08 ],
        #   [ 1.02726563e+08, 1.04776651e+08, 1.08121687e+08, 1.04799772e+08 ],
        #   [ 1.03927848e+08, 1.08360788e+08, 1.03373962e+08, 1.06481719e+08 ] ]
        , dtype=np.float64
        )
neg_memristors_np = np.array(
        [ [ 260289808, 94373320 ],
          [ -113619608, 264027088 ] ]
        # [ [ 1.03682415e+08, 1.09571552e+08, 1.01403508e+08, 1.08700873e+08 ],
        #   [ 1.04736080e+08, 1.08009108e+08, 1.05204775e+08, 1.06788795e+08 ],
        #   [ 1.07206327e+08, 1.05820198e+08, 1.05373732e+08, 1.07586156e+08 ],
        #   [ 1.01059076e+08, 1.04736004e+08, 1.01863323e+08, 1.07369182e+08 ] ]
        , dtype=np.float64
        )
r_min_np = np.array(
        [ [ 204.770325, 344.29834 ],
          [ 0.0795216262, 156.043961 ] ]
        # [ [ 281.09834969, 424.31067352, 435.48123994, 426.68877682 ],
        #   [ 295.38222438, 190.99956309, 153.96836575, 36.18402348 ],
        #   [ 143.12066054, 230.46580879, 400.51676371, 231.45276023 ],
        #   [ 194.54624384, 417.94761353, 171.08425145, 307.18294305 ] ]
        )
r_max_np = np.array(
        [ [ 2.35486e+08, 395943168 ],
          [ 91793.9922, 179450704 ] ]
        # [ [ 2.11805315e+08, 6.43659771e+08, 9.37120327e+07, 5.12958001e+08 ],
        #   [ 2.63046236e+08, 4.51738682e+08, 2.86222091e+08, 3.70826755e+08 ],
        #   [ 3.96137162e+08, 3.17590552e+08, 2.94706056e+08, 4.21048136e+08 ],
        #   [ 7.32425562e+07, 2.63042498e+08, 1.19478150e+08, 4.06563514e+08 ] ]
        
        )
exponent_np = np.array(
        [ [ 0.0911544263, -0.23531644 ],
          [ -0.223113075, -0.302653432 ] ]
        # [ [ -0.21085799, -0.19639718, -0.27474288, -0.21062737 ],
        #   [ -0.22497378, -0.3391912, -0.16246864, -0.01352749 ],
        #   [ -0.02697059, -0.1125517, -0.2958221, -0.07628128 ],
        #   [ 0.04273394, -0.2527929, -0.38038873, -0.00153648 ] ]
        )

local_error_tf = tf.reshape(
        tf.convert_to_tensor( local_error_np ),
        [ 1, 1, input_size, 1 ]
        )
pre_filtered_tf = tf.reshape(
        tf.convert_to_tensor( pre_filtered_np ),
        [ 1, 1, 1, input_size ]
        )
pos_memristors_tf = tf.reshape(
        tf.convert_to_tensor( pos_memristors_np ),
        [ 1, 1, output_size, input_size ]
        )
neg_memristors_tf = tf.reshape(
        tf.convert_to_tensor( neg_memristors_np ),
        [ 1, 1, output_size, input_size ]
        )
r_min_tf = tf.reshape(
        tf.convert_to_tensor( r_min_np ),
        [ 1, 1, output_size, input_size ]
        )
r_max_tf = tf.reshape(
        tf.convert_to_tensor( r_max_np ),
        [ 1, 1, output_size, input_size ]
        )
exponent_tf = tf.reshape(
        tf.convert_to_tensor( exponent_np ),
        [ 1, 1, output_size, input_size ]
        )

weights_np, pos_mem_np, neg_mem_np = np_calc( local_error_np,
                                              pre_filtered_np,
                                              copy.deepcopy( pos_memristors_np ),
                                              copy.deepcopy( neg_memristors_np ),
                                              r_min_np,
                                              r_max_np,
                                              exponent_np )
weights_tf, pos_mem_tf, neg_mem_tf = tf_calc( local_error_tf,
                                              pre_filtered_tf,
                                              pos_memristors_tf,
                                              neg_memristors_tf,
                                              r_min_tf,
                                              r_max_tf,
                                              exponent_tf )

print( "Memristors" )
print( "NumPy\n", pos_mem_np, "\n", neg_mem_np )
print( "TensorFlow\n", pos_mem_tf.numpy().squeeze(), "\n", neg_mem_tf.numpy().squeeze() )
print( "tf and np are equal?",
       np.array_equal( pos_mem_np, pos_mem_tf.numpy().squeeze() )
       and np.array_equal( neg_mem_np, neg_mem_tf.numpy().squeeze() )
       )
# print( "Before\n", pos_memristors_np, "\n", neg_memristors_np )
# print( "After\n", pos_mem_np, "\n", neg_mem_np )
# print( "before and after are equal?",
#        np.array_equal( pos_mem_np, pos_memristors_np )
#        and np.array_equal( neg_mem_np, neg_memristors_np )
#        )
print( "Weights" )
print( "NumPy\n", weights_np )
print( "TensorFlow\n", weights_tf.numpy().squeeze() )
print( "tf and np are equal?",
       np.array_equal( weights_np, weights_tf.numpy().squeeze() )
       )
