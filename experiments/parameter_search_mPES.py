import numpy as np
from subprocess import run
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle

from memristor_nengo.extras import *

# parameters to search
start_c = -0.0001
end_c = -1
num_c = 2
num_averaging = 2

c_list = np.linspace( start_c, end_c, num=num_c )
num_parameters = len( c_list )
print( "Number of parameters:", num_parameters )
print( "Averaging per parameter", num_averaging )
print( "Total iterations", num_parameters * num_averaging )

mse_list = [ ]
for k, c in enumerate( c_list ):
    print( f"Parameter #{k} ({c})" )
    it_res = [ ]
    for avg in range( num_averaging ):
        print( f"Averaging #{avg}" )
        result = run( [ "python", "mPES.py", "-v", "-d", "1", "-P", str( c ) ],
                      capture_output=True,
                      universal_newlines=True )
        # print( "Ret", result.returncode )
        # print( "Out", result.stdout )
        # print( "Err", result.stderr )
        # save MSE
        mse = float( result.stdout.split()[ 4 ] )
        print( mse )
        it_res.append( mse )
    mse_list.append( it_res )

dir_name, dir_images, dir_data = make_timestamped_dir( root="../data/parameter_search/" )

mse_means = np.mean( mse_list, axis=1 )
plt.plot( c_list, mse_means, label="MSE" )
exp_means_smooth = np.convolve( mse_means, np.ones( (mse_means.size,) ) / mse_means.size, mode="same" )
plt.plot( c_list, exp_means_smooth, label="MSE moving average" )

plt.legend()
plt.savefig( dir_images + "result" + ".pdf" )
plt.savefig( dir_images + "result" + ".png" )
plt.show()

pickle.dump( c_list, open( dir_data + "exponents" + ".pkl", "wb" ) )
pickle.dump( mse_list, open( dir_data + "mse" + ".pkl", "wb" ) )
