import subprocess
from functools import partial
import pickle
import nengo
from memristor_learning.Networks import *

print( "Pair, 1e-1, 1-constant VC" )
net = Network( memristor_controller=MemristorArray,
               memristor_model=partial( MemristorPair, model=MemristorAnouk, base_voltage=1e-1 ),
               voltage_converter=VoltageConverter )
net()

print( "Pair, 1e-10, 1-constant VC" )
net = Network( memristor_controller=MemristorArray,
               memristor_model=partial( MemristorPair, model=MemristorAnouk, base_voltage=1e-10 ),
               voltage_converter=VoltageConverter )
net()

print( "Pair, 1e-10, 10 level adaptive VC" )
net = Network( memristor_controller=MemristorArray,
               memristor_model=partial( MemristorPair, model=MemristorAnouk, base_voltage=1e-10 ),
               voltage_converter=LevelsVoltageConverter )
net()

print( "Bidirectional, 1e-10, 1-constant adaptive VC" )
net = Network( memristor_controller=MemristorArray,
               memristor_model=partial( MemristorAnoukBidirectional, base_voltage=1e-10 ),
               voltage_converter=VoltageConverter )
net()

print( "Bidirectional, 1e-10, 10 level adaptive VC" )
net = Network( memristor_controller=MemristorArray,
               memristor_model=partial( MemristorAnoukBidirectional, base_voltage=1e-10 ),
               voltage_converter=LevelsVoltageConverter )
net()
