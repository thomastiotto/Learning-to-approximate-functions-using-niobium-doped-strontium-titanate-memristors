from functools import partial
from memristor_learning.Networks import *

params = { "base voltage": 1e-1,
           "neurons"     : 4,
           "dimensions"  : 1 }

print( "\nPair, +0.1V" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorPlusMinus,
                                                   model=partial( MemristorAnouk ) ),
                          dimensions=params[ "dimensions" ],
                          # input_function=lambda t: np.sin( 1 / 30 * 2 * np.pi * t ),
                          # function_to_learn=lambda x: np.abs( x ),
                          # weight_modifier=ZeroShiftModifier,
                          base_voltage=params[ "base voltage" ],
                          seed=1,
                          weights_to_plot=[ 15 ],
                          neurons=params[ "neurons" ],
                          plot_ylim=(0, 2.5e-8),
                          smooth_plots=True,
                          input_function=lambda t: np.sin( 1 / 4 * 2 * np.pi * t )
                          )
res = net()

dir_name, dir_images = make_timestamped_dir( root="../data/models_test/mPlusMinus/" )
write_experiment_to_file( params, res,
                          [ "Memristor model", "Base V", "Neurons", "Input function", "Learned function" ],
                          [ [ "mPlusMinus", params[ "base voltage" ], params[ "neurons" ], "Sine", "Identity" ] ],
                          dir_name, dir_images )
