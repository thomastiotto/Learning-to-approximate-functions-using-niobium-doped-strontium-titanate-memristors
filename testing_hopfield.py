import nengo
from memristor_learning.MemristorHelpers import *
from memristor_learning.MemristorControllers import MemristorArray
from memristor_learning.MemristorModels import MemristorAnoukPair
from memristor_learning.MemristorLearningRules import mOja

# hyperparameters
neurons = 3
simulation_time = 30.0
learning_time = 15.0
simulation_step = 0.001

with nengo.Network() as model:
    inp = nengo.Node(
            [ 1, 0, 1 ]
            )
    nodes = nengo.Ensemble(
            n_neurons=neurons,
            dimensions=neurons,
            radius=1.5,
            label="Nodes"
            )
    error = nengo.Ensemble(
            n_neurons=10,
            dimensions=neurons,
            label="Error"
            )
    
    nengo.Connection( inp, nodes )
    
    inp_probe = nengo.Probe( inp, synapse=0.01 )
    nodes_probe = nengo.Probe( nodes, synapse=0.01 )

with nengo.Simulator( model, dt=simulation_step ) as sim:
    sim.run( simulation_time )

import matplotlib.pyplot as plt

plt.plot( sim.trange(), sim.data[ inp_probe ], c="k", label="Input" )
plt.show()
plt.plot( sim.trange(), sim.data[ nodes_probe ], c="k", label="Input" )
plt.show()
