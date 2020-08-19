import pickle
import time
from functools import partial
import os
import xarray as xr
from tabulate import tabulate

from memristor_learning.Networks import *

# parameters to search
start_r_0 = 1
end_r_0 = 4
num_r_0 = 10
start_r_1 = 5
end_r_1 = 9
num_r_1 = 10
start_a = -0.0001
end_a = -1
num_a = 1000

r_0_list = np.logspace( start_r_0, end_r_0, num=num_r_0 )
r_1_list = np.logspace( start_r_1, end_r_1, num=num_r_1 )
a_list = np.linspace( start_a, end_a, num=num_a )
total_iterations = num_a
assert (end_r_0 < start_r_1)
print( "Total averaging:", total_iterations )

dims = [ "exponent" ]
coords = dict.fromkeys( dims, 0 )
coords[ dims[ 0 ] ] = a_list

data = [ ]
results_dict = nested_dict( len( dims ), dict )

start_time = time.time()
curr_iteration = 0

for k, a in enumerate( a_list ):
    net = SupervisedLearning( memristor_controller=MemristorArray,
                              memristor_model=
                              partial( MemristorPlusMinus, model=
                              partial( OnedirectionalPowerlawMemristor, a=a, r_0=1e2, r_1=2.5e8 ) ),
                              seed=0,
                              neurons=4,
                              verbose=False,
                              generate_figures=False )
    res = net()
    print( res[ "mse" ] )
    data.append( res[ "mse" ] )
    results_dict[ a ] = res
    curr_iteration += 1
    print( f"{curr_iteration}/{total_iterations}:  {a}\n" )

# for i, x in enumerate( results ):
#     x[ "fig_pre_post" ].show()
#     time.sleep( 2 )
time_taken = time.time() - start_time
dir_name, dir_images = make_timestamped_dir( root="../data/parameter_search/mPlusMinus/" )
dataf = xr.DataArray( data=data, dims=dims, coords=coords, name="MSE" )
with open( f"{dir_name}mse.pkl", "wb" ) as f:
    pickle.dump( dataf, f )
table = [ [ "exponent", start_a, end_a, num_a ],
          ]
headers = [ "Parameter", "Start", "End", "Number" ]
with open( f"{dir_name}param.txt", "w+" ) as f:
    f.write( tabulate( table, headers=headers, tablefmt="github" ) )
    f.write( f"\n\nTotal time: {datetime.timedelta( seconds=time_taken )}" )
    f.write( f"\nTime per iteration: {round( time_taken / total_iterations, 2 )} s" )
    
    # loaded = pickle.load( open( f"{dir_name}mse.pkl", "rb" ) )
