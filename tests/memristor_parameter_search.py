from functools import partial
from matplotlib import pyplot as plt
from memristor_learning.Networks import *
import pickle

results = [ ]

# parameters
a_list = np.linspace( -0.1, -0.3, num=5 )
c_list = np.linspace( -0.1, -0.3, num=5 )

for a in a_list:
    print( a )
    net = SupervisedLearning( memristor_controller=MemristorArray,
                              memristor_model=
                              partial( MemristorPair, model=
                              partial( MemristorAnouk, a=a, b=0 ) ),
                              seed=0,
                              weights_to_plot=[ 15 ],
                              neurons=2,
                              verbose=False )
    results.append( net() )
