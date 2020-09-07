import argparse
from subprocess import run

from memristor_nengo.extras import *

parser = argparse.ArgumentParser()
parser.add_argument( "-a", "--averaging", type=int, required=True )
parser.add_argument( "-N", "--neurons", type=int, required=True )
parser.add_argument( "-D", "--dimensions", type=int, required=True )
parser.add_argument( "-l", "--learning_rule", choices=[ "mPES", "PES" ], required=True )
parser.add_argument( "-d", "--directory", default="../data/" )
args = parser.parse_args()

learning_rule = args.learning_rule
neurons = args.neurons
dimensions = args.dimensions
num_averaging = args.averaging
directory = args.directory

print( "Evaluation for", learning_rule )
print( "Averaging per rule", num_averaging )
print( "Total iterations", num_averaging )

mse_res = [ ]
for avg in range( num_averaging ):
    print( f"Averaging #{avg}" )
    result = run(
            [ "python", "mPES.py", "-v", "-d", str( dimensions ), "-l", str( learning_rule ), "-N", str( neurons ) ],
            capture_output=True,
            universal_newlines=True )
    # print( "Ret", result.returncode )
    # print( "Out", result.stdout )
    # print( "Err", result.stderr )
    # save MSE
    mse = float( result.stdout.split()[ 4 ] )
    print( mse )
    mse_res.append( mse )

dir_name, dir_images, dir_data = make_timestamped_dir( root=directory + "averaging/" + str( learning_rule ) + "/" )

mse_means = np.mean( mse_res )
plt.plot( range( num_averaging ), mse_res, label="MSE" )
plt.axhline( mse_means, color="r", label="MSE average" )
plt.annotate( str( mse_means ), xy=(0, mse_means) )

plt.legend()
plt.savefig( dir_images + "result" + ".pdf" )
plt.savefig( dir_images + "result" + ".png" )
plt.show()

# pickle.dump( res_list, open( dir_data + str( parameter ) + ".pkl", "wb" ) )
# pickle.dump( mse_list, open( dir_data + "mse" + ".pkl", "wb" ) )
np.savetxt( dir_data + "results.csv",
            mse_res,
            delimiter=",", header="MSE", comments="" )
