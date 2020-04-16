from functools import partial
from memristor_learning.Networks import *

print( "\nPair, +0.1V" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorPair, model=MemristorAnouk ),
                          voltage_converter=VoltageConverter,
                          base_voltage=1e-1,
                          seed=0,
                          weights_to_plot=[ 15 ],
                          neurons=2 )
net()

print( "\nBidirectional, +0.1V, c=-0.001, d=0.0" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorAnoukBidirectional, c=-0.01, d=0.0 ),
                          voltage_converter=VoltageConverter,
                          base_voltage=1e-1,
                          seed=0,
                          weights_to_plot=[ 15 ],
                          neurons=2 )
net()
