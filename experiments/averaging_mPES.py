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

dir_name, dir_images, dir_data = make_timestamped_dir(
        root=directory + "averaging/" + str( learning_rule ) + "/" + function + "_" + input + "_" + str( neurons ) + "_"
             + str( dimensions ) + "_" + str( gain ) + "/" )
print( "Reserved folder", dir_name )

print( "Evaluation for", learning_rule )
print( "Averaging runs", num_averaging )

res_mse = [ ]
res_pearson = [ ]
res_spearman = [ ]
res_kendall = [ ]
counter = 0
for avg in range( num_averaging ):
    counter += 1
    print( f"[{counter}/{num_averaging}] Averaging #{avg}" )
    result = run(
            [ "python", "mPES.py", "-v", "-D", str( dimensions ), "-l", str( learning_rule ), "-N", str( neurons ),
              "-f", str( function ), "-i", str( input ), "-lt", str( learn_time ), "-g", str( gain ) ],
            capture_output=True,
            universal_newlines=True )
    
    # save statistics
    try:
        mse = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 0 ][ 1:-1 ].split( "," ) ] )
        print( "MSE", mse )
        res_mse.append( mse )
        pearson = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 1 ][ 1:-1 ].split( "," ) ] )
        print( "Pearson", pearson )
        res_pearson.append( pearson )
        spearman = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 2 ][ 1:-1 ].split( "," ) ] )
        print( "Spearman", spearman )
        res_spearman.append( spearman )
        kendall = np.mean( [ float( i ) for i in result.stdout.split( "\n" )[ 3 ][ 1:-1 ].split( "," ) ] )
        print( "Kendall", kendall )
        res_kendall.append( kendall )
    except:
        print( "Ret", result.returncode )
        print( "Out", result.stdout )
        print( "Err", result.stderr )
mse_means = np.mean( res_mse )
pearson_means = np.mean( res_pearson )
spearman_means = np.mean( res_spearman )
kendall_means = np.mean( res_kendall )
print( "Average MSE:", mse_means )
print( "Average Pearson:", pearson_means )
print( "Average Spearman:", spearman_means )
print( "Average Kendall:", kendall_means )

res_list = range( num_averaging )

fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.plot( res_list, res_mse, label="MSE" )
ax.legend()
fig.savefig( dir_images + "mse" + ".pdf" )

fig = plt.figure()
ax = fig.add_subplot( 111 )
ax.plot( res_list, res_pearson, label="Pearson" )
ax.plot( res_list, res_spearman, label="Spearman" )
ax.plot( res_list, res_kendall, label="Kendall" )
ax.legend()
fig.savefig( dir_images + "correlations" + ".pdf" )

print( f"Saved plots in {dir_images}" )

np.savetxt( dir_data + "results.csv",
            np.stack( (res_mse, res_pearson, res_spearman, res_kendall), axis=1 ),
            delimiter=",", header="MSE,Pearson,Spearman,Kendall", comments="" )
with open( dir_data + "parameters.txt", "w" ) as f:
    f.write( f"Learning rule: {learning_rule}\n" )
    f.write( f"Function: {function}\n" )
    f.write( f"Neurons: {neurons}\n" )
    f.write( f"Dimensions: {dimensions}\n" )
    f.write( f"Number of runs for averaging: {num_averaging}\n" )
print( f"Saved data in {dir_data}" )
