import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import nengo_dl
from nengo.processes import WhiteNoise, WhiteSignal
from nengo.dists import Gaussian
from nengo.learning_rules import PES

from memristor_nengo.extras import *
from memristor_nengo.learning_rules import mPES

learn_time = 40
timestep = 0.001

model = nengo.Network()
with model:
    # -- input and pre population
    inp = nengo.Node(
            # WhiteNoise( dist=Gaussian( 0, 0.05 ) ),
            WhiteSignal( 60, 5 ),
            size_out=2
            )
    pre = nengo.Ensemble( 200, dimensions=2 )
    nengo.Connection( inp, pre )
    
    # -- post population
    post = nengo.Ensemble( 200, dimensions=1 )
    
    
    # -- reference population, containing the actual product
    def function_to_learn( x ):
        return x[ 0 ] * x[ 1 ]
    
    
    product = nengo.networks.Product( n_neurons=100, dimensions=1, input_magnitude=1 )
    nengo.Connection( inp[ 0 ], product.input_a )
    nengo.Connection( inp[ 1 ], product.input_b )
    
    # -- error population
    error = nengo.Ensemble( 100, dimensions=1 )
    nengo.Connection( post, error )
    nengo.Connection( product.output, error, transform=-1 )
    
    # -- learning connection
    conn = nengo.Connection(
            pre.neurons,
            post.neurons,
            transform=np.zeros( (post.n_neurons, pre.n_neurons) ),
            learning_rule_type=mPES( gain=1e3 ),
            # learning_rule_type=PES(),
            )
    
    nengo.Connection( error, conn.learning_rule )
    
    # -- inhibit error after 40 seconds
    inhib = nengo.Node( lambda t: 2.0 if t > 40.0 else 0.0 )
    nengo.Connection( inhib, error.neurons, transform=[ [ -1 ] ] * error.n_neurons )
    
    # -- probes
    product_probe = nengo.Probe( product.output, synapse=0.01 )
    pre_probe = nengo.Probe( pre, synapse=0.01 )
    post_probe = nengo.Probe( post, synapse=0.01 )
    error_probe = nengo.Probe( error, synapse=0.03 )

with nengo_dl.Simulator( model ) as sim:
    sim.run( 60 )

# plt.figure( figsize=(12, 8) )
# plt.subplot( 3, 1, 1 )
# plt.plot( sim.trange(), sim.data[ pre_probe ], c="b" )
# plt.legend( ("Pre decoding",), loc="best" )
# plt.subplot( 3, 1, 2 )
# plt.plot( sim.trange(), sim.data[ product_probe ], c="k", label="Actual product" )
# plt.plot( sim.trange(), sim.data[ post_probe ], c="r", label="Post decoding" )
# plt.legend( loc="best" )
# plt.subplot( 3, 1, 3 )
# plt.plot( sim.trange(), sim.data[ error_probe ], c="b" )
# plt.ylim( -1, 1 )
# plt.legend( ("Error",), loc="best" )
# plt.show()

plt.figure( figsize=(12, 8) )
plt.subplot( 2, 1, 1 )
plt.plot( sim.trange()[ int( (learn_time / timestep) ): ], sim.data[ pre_probe ][ int( (learn_time / timestep) ): ],
          c="b" )
plt.legend( ("Pre decoding",), loc="best" )
plt.subplot( 2, 1, 2 )
plt.plot(
        sim.trange()[ int( (learn_time / timestep) ): ],
        sim.data[ product_probe ][ int( (learn_time / timestep) ): ],
        c="k",
        label="Actual product",
        )
plt.plot(
        sim.trange()[ int( (learn_time / timestep) ): ],
        sim.data[ post_probe ][ int( (learn_time / timestep) ): ],
        c="r",
        label="Post decoding",
        )
plt.legend( loc="best" )
plt.ylim( -1, 1 )
plt.legend( ("Error",), loc="best" )
plt.show()

# essential statistics
y_pred = sim.data[ post_probe ][ int( (learn_time / timestep) ):, ... ]
y_true = sim.data[ product_probe ][ int( (learn_time / timestep) ):, ... ]
# MSE after learning
print( "MSE after learning [f(pre) vs. post]:" )
mse = mean_squared_error( y_true, y_pred, multioutput='raw_values' )
print( mse.tolist() )

# st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
