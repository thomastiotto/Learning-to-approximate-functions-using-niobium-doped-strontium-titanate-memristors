import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import csv

r_min = 1e2
r_max = 2.5e8
g_min = 1 / r_max
g_max = 1 / r_min
gain = 1e5
a = -0.128

iterations = 100000


def monom_deriv( base, exp ):
    return exp * base**(exp - 1)


def resistance2conductance( R ):
    g_curr = 1.0 / R
    g_norm = (g_curr - g_min) / (g_max - g_min)
    
    # return gain * (g_norm + epsilon)
    # return g_norm * gain
    return g_norm


def conductance2resistance( G ):
    # g_clean = (G / gain) - epsilon
    # g_unnorm = g_clean * (g_max - g_min) + g_min
    g_unnorm = G * (g_max - g_min) + g_min
    
    return 1.0 / g_unnorm


n = [ ]
r = [ 1e8 ]
g = [ ]

for i in range( iterations ):
    n.append( ((r[ -1 ] - r_min) / r_max)**(1 / a) )
    r.append( r_min + r_max * (n[ -1 ] + 1)**a )
    g.append( resistance2conductance( r[ -1 ] ) )

n = np.array( n )
r = np.array( r )
g = np.array( g )


def fit_func_exp( x, a ):
    return g_min * x**(a)


def fit_func_poly( x, *coeffs ):
    return np.polyval( coeffs, x )


params_exp, params_exp_covariance = curve_fit( fit_func_exp, n, g )
g_test_exp = fit_func_exp( n, *params_exp )

plt.figure()
plt.plot( n, g, label="Ground truth" )
plt.plot( n, g_test_exp, label="Exp fit" )
plt.xlabel( "Pulse number" )
plt.ylabel( "Conductance" )
plt.legend()
plt.show()

fit_results = [ ]
for p in range( 2, 10 ):
    p0 = np.ones( p )
    
    popt, pcov = curve_fit( fit_func_poly, n, g, p0=p0 )
    fit_results.append( popt )

plt.plot( n, g, 'k.', label='Ground truth' )

xx = np.linspace( n.min(), n.max(), 100 )
for p in fit_results:
    yy = fit_func_poly( xx, *p )
    plt.plot( xx, yy, alpha=0.6, label='n = %d' % len( p ) )

plt.legend( framealpha=1, shadow=True )
plt.xlabel( 'Pulse number' )
plt.ylabel( "Conductance" )
plt.show()

rows = zip( r, n, g, g_test_exp )
with open( "../data/conductance_fit.csv", "w" ) as f:
    writer = csv.writer( f )
    writer.writerow(
            [ "Resistance (R)", "Pulse (x)", "Ground truth (y)", "Exponents fit (y_hat=R_0+R_1*x**exponent)" ] )
    for row in rows:
        writer.writerow( row )

# print( "Difference in conductances:" )
# print( np.abs( np.array( g ) - np.array( test_g ) ) )
# print( "Average difference in conductances:" )
# avg_err = np.sum( np.abs( np.array( g ) - np.array( test_g ) ) ) / len( r )
# print( avg_err, f"({(avg_err * 100) / (g_max - g_min)} %)" )
# print( "MSE in conductances:" )
# print( mean_squared_error( g, test_g ) )
# print( "Parameters:" )
# print( params )
