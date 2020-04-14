from functools import partial
from memristor_learning.Experiments import *

print( "\nPair, +0.01V, c=-0.01, d=0.0" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorPair, model=MemristorAnouk ),
                          voltage_converter=VoltageConverter,
                          base_voltage=1e-10,
                          seed=0,
                          weights_to_plot=[ 15 ] )
net()

print( "\nBidirectional, +0.01V, c=-0.01, d=0.0" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorAnoukBidirectional, c=-0.01, d=0.0 ),
                          voltage_converter=VoltageConverter,
                          base_voltage=1e-10,
                          seed=0,
                          weights_to_plot=[ 15 ] )
net()
