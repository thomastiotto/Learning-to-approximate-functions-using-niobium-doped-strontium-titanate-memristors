import numpy as np
from sklearn.metrics import mean_squared_error

r_min = 1e2
r_max = 2.5e8
norm_constant = 5e7
g_min = 1 / r_max
g_max = 1 / r_min
gain = 1e5
a = -0.1

iterations = 100000


def monom_deriv( base, exp ):
    return exp * base**(exp - 1)


def resistance2conductance( R ):
    g_curr = 1 / R
    g_norm = (g_curr - g_min) / (g_max - g_min)
    
    # return g_curr
    return g_norm
    # return g_norm


def conductance2resistance( G ):
    # g_clean = (G / gain) - epsilon
    # g_unnorm = g_clean * (g_max - g_min) + g_min
    g_unnorm = G * (g_max - g_min) + g_min
    
    return 1.0 / g_unnorm


r_init = 1e8
g_init = resistance2conductance( r_init )

n = [ ]
r = [ r_init ]
g = [ g_init ]
g_delta = [ ]
r_delta = [ ]

for i in range( iterations ):
    n.append( ((r[ -1 ] - r_min) / r_max)**(1 / a) )
    r_delta.append( r_max * monom_deriv( n[ -1 ], a ) )
    r.append( r[ -1 ] + r_delta[ -1 ] )
    g.append( resistance2conductance( r[ -1 ] ) )
    # g_delta.append( resistance2conductance( r[ -1 ] + r_delta[ -1 ] ) - resistance2conductance( r[ -1 ] ) )

n_2 = [ ]
g_2 = [ g_init ]
g_delta_2 = [ ]

for i in range( iterations ):
    n_2.append( ((g_2[ -1 ] - g_max) * r_max)**(1 / a) )
    g_delta_2.append( g_max * monom_deriv( n[ -1 ], a ) )
    g_2.append( g_2[ -1 ] + g_delta_2[ -1 ] )

n_3 = [ ]
g_3 = [ g_init ]
g_delta_3 = [ ]
for i in range( iterations ):
    n_3.append( ((1 / g_3[ -1 ] - 1 / g_max) * g_min)**(1 / a) )
    g_delta_3.append( -1 * g_max * g_min * (g_min + g_max * n_3[ -1 ]**a)**(-2) * g_max * monom_deriv( n_3[ -1 ], a ) )
    g_3.append( g_3[ -1 ] + g_delta_3[ -1 ] )

print( "Difference in conductances:" )
print( np.abs( np.array( g ) - np.array( g_3 ) ) )
print( "Average difference in conductances:" )
avg_err = np.sum( np.abs( np.array( g ) - np.array( g_3 ) ) ) / len( r )
print( avg_err, f"({(avg_err * 100) / (resistance2conductance( r_min ) - resistance2conductance( r_max ))} %)" )
print( "MSE in conductances:" )
print( mean_squared_error( g, g_3 ) )
