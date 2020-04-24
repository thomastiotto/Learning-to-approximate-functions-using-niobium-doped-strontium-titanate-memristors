from functools import partial
from memristor_learning.Networks import *

print( "\nPair, +0.1V" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( MemristorPair, model=partial( MemristorAnouk ) ),
                          # input_function=lambda t: np.sin( 1 / 30 * 2 * np.pi * t ),
                          weight_modifier=ZeroShiftModifier,
                          base_voltage=1e-1,
                          seed=0,
                          weights_to_plot=[ 15 ],
                          neurons=4
                          )
res = net()

dir_name, dir_images = make_timestamped_dir( root="../data/learning_test/mBi/" )
write_experiment_to_file( res,
                          [ "Memristor model", "Base V", "Neurons", "Input function", "Learned function" ],
                          [ [ "mPair", 0.1, 4, "Sine", "Identity" ] ],
                          dir_name, dir_images )
