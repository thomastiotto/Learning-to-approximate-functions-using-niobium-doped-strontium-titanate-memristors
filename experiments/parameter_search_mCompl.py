import pickle
import time
from functools import partial
import os
import xarray as xr
from tabulate import tabulate

from memristor_learning.Networks import *

# parameters to search
start_a = -0.0001
end_a = -1
num_a = 30
start_c = -0.0001
end_c = -1
num_c = 30

a_list = np.linspace( start_a, end_a, num=num_a )
c_list = np.linspace( start_c, end_c, num=num_c )
total_iterations = num_a * num_c
print( "Total averaging:", total_iterations )

dims = [ "exponent", "c" ]
coords = dict.fromkeys( dims, 0 )
coords[ dims[ 0 ] ] = a_list
coords[ dims[ 1 ] ] = c_list

data = [ ]
results_dict = nested_dict( len( dims ), dict )

start_time = time.time()
curr_iteration = 0
for i, a in enumerate( a_list ):
    data.append( [ ] )
    for j, c in enumerate( c_list ):
        net = SupervisedLearning( memristor_controller=MemristorArray,
                                  memristor_model=
                                  partial( MemristorPlusMinus, model=
                                  partial( BidirectionalPowerlawMemristor, a=-0.223, c=-0.001, r_0=1e2, r_1=2.5e8 ) ),
                                  seed=0,
                                  neurons=4,
                                  verbose=False,
                                  generate_figures=False )
        res = net()
        print( res[ "mse" ] )
        data[ i ].append( res[ "mse" ] )
        results_dict[ a ][ c ] = res
        curr_iteration += 1
        print( f"{curr_iteration}/{total_iterations}: {a}, {c}\n" )

time_taken = time.time() - start_time
dir_name, dir_images = make_timestamped_dir( root="../data/parameter_search/mCompl/" )
dataf = xr.DataArray( data=data, dims=dims, coords=coords )
with open( f"{dir_name}mse.pkl", "wb" ) as f:
    pickle.dump( dataf, f )
table = [ [ "exponent", start_a, end_a, num_a ],
          [ "c", start_c, end_c, num_c ] ]
headers = [ "Parameter", "Start", "End", "Number" ]
with open( f"{dir_name}param.txt", "w+" ) as f:
    f.write( tabulate( table, headers=headers, tablefmt="github" ) )
    f.write( f"\n\nTotal time: {datetime.timedelta( seconds=time_taken )}" )
    f.write( f"\nTime per iteration: {round( time_taken / total_iterations, 2 )} s" )
    
    # loaded = pickle.load( open( f"{dir_name}mse.pkl", "rb" ) )
