from functools import partial
from memristor_learning.Networks import *

params = { "c"           : -0.001,
           "d"           : 0,
           "base voltage": 1e-1,
           "neurons"     : 4 }

net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorComplementary,
                                                   model=partial( MemristorAnoukBidirectional, c=params[ "c" ],
                                                                  d=params[ "d" ] ) ),
                          # input_function=lambda t: np.sin( 1 / 30 * 2 * np.pi * t ),
                          # weight_modifier=ZeroShiftModifier,
                          base_voltage=params[ "base voltage" ],
                          seed=0,
                          weights_to_plot=[ 15 ],
                          neurons=params[ "neurons" ],
                          plot_ylim=(0, 2.5e-8)
                          )
res = net()

dir_name, dir_images = make_timestamped_dir( root="../data/models_test/mCompl/" )
write_experiment_to_file( params, res,
                          [ "Memristor model", "Base V", "c", "d", "Neurons", "Input function", "Learned function" ],
                          [ [ "mCompl", params[ "base voltage" ], params[ "c" ], params[ "d" ], "Sine", "Identity" ] ],
                          dir_name, dir_images )
