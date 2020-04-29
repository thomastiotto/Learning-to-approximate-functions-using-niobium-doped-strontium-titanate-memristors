from functools import partial
from memristor_learning.Networks import *

print( "\nBidirectional, +0.1V, c=-0.001, d=0.0" )
net = SupervisedLearning( memristor_controller=MemristorArray,
                          memristor_model=partial( BidirectionalPowerlawMemristor, c=-0.001, d=0.0 ),
                          # input_function=lambda t: np.sin( 1 / 30 * 2 * np.pi * t ),
                          weight_modifier=ZeroShiftModifier,
                          base_voltage=1e-2,
                          seed=0,
                          weights_to_plot=[ 15 ],
                          neurons=4
                          )
res = net()

dir_name, dir_images = make_timestamped_dir( root="../data/learning_test/mPair/" )
write_experiment_to_file( res,
                          [ "Memristor model", "Base V", "c parameter", "Neurons", "Input function",
                            "Learned function" ],
                          [ [ "mPair", 0.01, 0.001, 4, "Sine", "Identity" ] ],
                          dir_name, dir_images )
# TODO write paramters to file
