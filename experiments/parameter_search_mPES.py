import argparse
from subprocess import run

from memristor_nengo.extras import *

parser = argparse.ArgumentParser()
parser.add_argument( "-p", "--parameter", choices=[ "exponent", "noise", "neurons", "gain" ], required=True )
parser.add_argument( "-l", "--limits", nargs=2, type=float, required=True )
parser.add_argument( "-n", "--number", type=int )
parser.add_argument( "-a", "--averaging", type=int, required=True )
parser.add_argument( "-d", "--directory", default="../data/" )
args = parser.parse_args()
# parameters to search
parameter = args.parameter
start_par = args.limits[ 0 ]
end_par = args.limits[ 1 ]
num_par = args.number if args.parameter in [ "exponent", "noise", "neurons" ] else end_par - start_par + 1
num_averaging = args.averaging
directory = args.directory

res_list = np.linspace( start_par, end_par, num=num_par ) if args.parameter in [ "exponent", "noise", "neurons" ] \
    else np.logspace( np.rint( start_par ).astype( int ), np.rint( end_par ).astype( int ),
                      num=np.rint( num_par ).astype( int ) )
num_parameters = len( res_list )
print( "Evaluation for", parameter )
print( f"Search limits of parameters: [{start_par},{end_par}]" )
print( "Number of parameters:", num_parameters )
print( "Averaging per parameter", num_averaging )
print( "Total iterations", num_parameters * num_averaging )

mse_list = [ ]
for k, par in enumerate( res_list ):
    print( f"Parameter #{k} ({par})" )
    it_res = [ ]
    for avg in range( num_averaging ):
        print( f"Averaging #{avg}" )
        if parameter == "exponent":
            result = run( [ "python", "mPES.py", "-v", "-D", "1", "-P", str( par ) ],
                          capture_output=True,
                          universal_newlines=True )
        if parameter == "noise":
            result = run( [ "python", "mPES.py", "-v", "-D", "1", "-n", str( par ) ],
                          capture_output=True,
                          universal_newlines=True )
        if parameter == "neurons":
            rounded_neurons = str( np.rint( par ).astype( int ) )
            result = run( [ "python", "mPES.py", "-v", "-D", "1",
                            "-N", str( 100 ), rounded_neurons, str( 100 ) ],
                          capture_output=True,
                          universal_newlines=True )
        if parameter == "gain":
            result = run( [ "python", "mPES.py", "-v", "-D", "1", "-g", str( par ) ],
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

dir_name, dir_images, dir_data = make_timestamped_dir( root=directory + "parameter_search/" + str( parameter ) + "/" )

mse_means = np.mean( mse_list, axis=1 )
plt.plot( res_list, mse_means, label="MSE" )
mse_means_smooth = np.convolve( mse_means, np.ones( (mse_means.size,) ) / mse_means.size, mode="same" )
plt.plot( res_list, mse_means_smooth, label="MSE average" )

plt.legend()
plt.savefig( dir_images + "result" + ".pdf" )
plt.savefig( dir_images + "result" + ".png" )
# plt.show()

# pickle.dump( res_list, open( dir_data + str( parameter ) + ".pkl", "wb" ) )
# pickle.dump( mse_list, open( dir_data + "mse" + ".pkl", "wb" ) )
np.savetxt( dir_data + "results.csv",
            np.stack( (res_list, mse_means, mse_means_smooth), axis=1 ),
            delimiter=",", header=parameter + ",MSE,MSE smooth", comments="" )
