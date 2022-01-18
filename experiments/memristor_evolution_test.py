import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

r_min = 2e2
r_max = 2.3e8
a = -0.146

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

size_L=10
size_M=8
size_S=6
fig, ax = plt.subplots()
fig.set_size_inches( (3.5, 3.5*((5.**0.5-1.0)/2.0)) )
plt.title( 'Memristor resistance',fontsize=size_L )
ax.set_xlabel( r"Pulse number $n$", fontsize=size_M )
ax.set_ylabel( r"Resistance $R (\Omega)$", fontsize=size_M )
ax.tick_params(axis='x', labelsize=size_S)
ax.tick_params(axis='y', labelsize=size_S)
plt.ticklabel_format(axis='both', style='scientific')
ax.plot(n_2,r_2[:-1])

# ax2 = plt.axes([0,0,1,1])
# ip = InsetPosition(ax, [0.4,0.2,0.5,0.5])
# ax2.set_axes_locator(ip)
# mark_inset(ax, ax2, loc1=2, loc2=4, fc="none", ec='0.5')
# ax2.plot(n_2[1:4],r_2[1:4])
plt.tight_layout()
# ax.plot(n,np.diff(r))
fig.show()
fig.savefig( "resistance.pdf" )

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
