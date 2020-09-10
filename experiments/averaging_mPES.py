import argparse
from subprocess import run

from memristor_nengo.extras import *

parser = argparse.ArgumentParser()
parser.add_argument( "-a", "--averaging", type=int, required=True )
parser.add_argument( "-i", "--input", choices=[ "sine", "white" ], required=True )
parser.add_argument( "-f", "--function", default="x" )
parser.add_argument( "-N", "--neurons", type=int, default=10 )
parser.add_argument( "-D", "--dimensions", type=int, default=3 )
parser.add_argument( "-g", "--gain", type=float, default=1e5 )
parser.add_argument( "-l", "--learning_rule", choices=[ "mPES", "PES" ], required=True )
parser.add_argument( "-d", "--directory", default="../data/" )
parser.add_argument( "-lt", "--learn_time", default=3 / 4, type=float )
args = parser.parse_args()

learning_rule = args.learning_rule
gain = args.gain
function = args.function
input = args.input
neurons = args.neurons
dimensions = args.dimensions
num_averaging = args.averaging
directory = args.directory
learn_time = args.learn_time

print( "Evaluation for", learning_rule )
print( "Averaging runs", num_averaging )

mse_res = [ ]
for avg in range( num_averaging ):
    print( f"Averaging #{avg}" )
    result = run(
            [ "python", "mPES.py", "-v", "-D", str( dimensions ), "-l", str( learning_rule ), "-N", str( neurons ),
              "-f", str( function ), "-i", str( input ), "-lt", str( learn_time ), "-g", str( gain ) ],
            capture_output=True,
            universal_newlines=True )
    print( "Ret", result.returncode )
    print( "Out", result.stdout )
    print( "Err", result.stderr )
    # save MSE
    mse = float( result.stdout.split()[ 4 ] )
    print( mse )
    mse_res.append( mse )

dir_name, dir_images, dir_data = make_timestamped_dir(
        root=directory + "averaging/" + function + "/" + str( learning_rule ) + "/" )

mse_means = np.mean( mse_res )
print( "Average MSE:", mse_means )
plt.plot( range( num_averaging ), mse_res, label="MSE" )
plt.axhline( mse_means, color="r", label="MSE average" )
plt.annotate( str( mse_means ), xy=(0, mse_means) )

plt.legend()
plt.savefig( dir_images + "result" + ".pdf" )
plt.savefig( dir_images + "result" + ".png" )
# plt.show()

np.savetxt( dir_data + "results.csv",
            mse_res,
            delimiter=",", header="MSE", comments="" )
with open( dir_data + "parameters.txt", "w" ) as f:
    f.write( f"Learning rule: {learning_rule}\n" )
    f.write( f"Function: {function}\n" )
    f.write( f"Neurons: {neurons}\n" )
    f.write( f"Dimensions: {dimensions}\n" )
    f.write( f"Number of runs for averaging: {num_averaging}\n" )
