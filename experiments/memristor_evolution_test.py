import numpy as np
from sklearn.metrics import mean_squared_error as mse

r_min = 1e2
r_max = 2.5e8
a = -0.128

iterations = 100000


def monom_deriv( base, exp ):
    return exp * base**(exp - 1)


n = [ ]
r = [ 1e8 ]
r_delta = [ ]

for i in range( iterations ):
    n.append( ((r[ -1 ] - r_min) / r_max)**(1 / a) )
    r_delta.append( r_max * monom_deriv( n[ -1 ], a ) )
    r.append( r[ -1 ] + r_delta[ -1 ] )

print( "Derivative method:" )
print( n )
print( r )
print( r_delta )
print( np.diff( r_delta ) )

n_2 = [ ]
r_2 = [ 1e8 ]
r_delta_2 = [ ]

for i in range( iterations ):
    n_2.append( ((r_2[ -1 ] - r_min) / r_max)**(1 / a) )
    r_2.append( r_min + r_max * (n_2[ -1 ] + 1)**a )

print( "Direct method:" )
print( n_2 )
print( r_2 )
print( np.diff( r_2 ) )

print( "Difference in resistances:" )
print( np.abs( np.array( r ) - np.array( r_2 ) ) )
print( "Average difference in resistances:" )
avg_err = np.sum( np.abs( np.array( r ) - np.array( r_2 ) ) ) / len( r )
print( avg_err, f"({(avg_err * 100) / (r_max - r_min)} %)" )
print( "MSE in resistances:" )
print( mse( r, r_2 ) )

# delta[ V > 0 ] = resistance2conductance( pos_memristors[ V > 0 ] + pos_update ) \
#                  - resistance2conductance( pos_memristors[ V > 0 ] )
#
# pos_memristors[ V > 0 ] += pos_update
