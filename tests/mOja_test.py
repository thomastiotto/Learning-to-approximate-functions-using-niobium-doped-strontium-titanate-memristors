import nengo
import numpy as np
from memristor_nengo.extras import *
from memristor_nengo.learning_rules import mOja, mPES
from memristor_nengo.neurons import AdaptiveLIFLateralInhibition

seed = 42
beta = 1

model = nengo.Network( seed=seed )
with model:
    inp = nengo.Node( [ 2, 2, -2, -2 ] )
    pre = nengo.Ensemble( 4, 1,
                          # neuron_type=AdaptiveLIFLateralInhibition(),
                          seed=seed )
    nengo.Connection( inp, pre.neurons, seed=seed )
    conn = nengo.Connection( pre.neurons, pre.neurons,
                             # learning_rule_type=nengo.learning_rules.Oja( beta=beta ),
                             learning_rule_type=mOja( beta=beta, noisy=0.01 ),
                             # transform=np.random.random( (pre.n_neurons, pre.n_neurons) ),
                             transform=np.zeros( (pre.n_neurons, pre.n_neurons) ),
                             # transform=np.eye( pre.n_neurons ),
                             seed=seed
                             )
    # conn_inh = nengo.Connection( pre.neurons, pre.neurons,
    #                              transform=-2 * (np.ones( (pre.n_neurons, pre.n_neurons) ) - np.eye( pre.n_neurons )),
    #                              seed=seed
    #                              )
    
    pre_probe = nengo.Probe( pre.neurons )
    weight_probe = nengo.Probe( conn, "weights" )

with nengo.Simulator( model, seed=seed ) as sim:
    sim.run( 10 )

fig1 = neural_activity_plot( sim.data[ pre_probe ], sim.trange() )
fig1.show()
plt.set_cmap( 'jet' )
fig2, ax = plt.subplots( 1 )
ax.matshow( sim.data[ weight_probe ][ -1 ] )
fig2.show()
print()
print( sim.data[ weight_probe ][ -1 ] )
