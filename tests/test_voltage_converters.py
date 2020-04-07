import subprocess
from functools import partial
import pickle
import nengo
from memristor_learning.Experiments import *

print( "\nPair, 1e-1, 1-constant VC" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorPair, model=MemristorAnouk, base_voltage=1e-1 ),
                          voltage_converter=VoltageConverter,
                          seed=0 )
net()

print( "\nPair, 1e-10, 1-constant VC" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorPair, model=MemristorAnouk, base_voltage=1e-10 ),
                          voltage_converter=VoltageConverter,
                          seed=0 )
net()

print( "\nPair, 1e-10, 10-level adaptive VC" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorPair, model=MemristorAnouk, base_voltage=1e-10 ),
                          voltage_converter=LevelsVoltageConverter,
                          seed=0 )
net()

print( "\nBidirectional, 1e-10, 1-constant VC" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorAnoukBidirectional, base_voltage=1e-10 ),
                          voltage_converter=VoltageConverter,
                          seed=0 )
net()

print( "\nBidirectional, 1e-10, 10-level adaptive VC" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorAnoukBidirectional, base_voltage=1e-10 ),
                          voltage_converter=partial( LevelsVoltageConverter, levels=100 ),
                          seed=0 )
net()
