from memristor_nengo.extras import *

plots = [ ]
plotter = Plotter( sim.trange( sample_every=sample_every ), post.n_neurons, pre.n_neurons, dimensions, learn_time,
                   plot_size=(13, 7),
                   dpi=72 )

plots.append(
        plotter.plot_results_no_input( sim.data[ input_node_probe ], sim.data[ pre_probe ], sim.data[ post_probe ],
                                       error=sim.data[ post_probe ] - function_to_learn( sim.data[ pre_probe ] ),
                                       smooth=True,
                                       mse=mse )
        )
plots.append(
        plotter.plot_ensemble_spikes( "Post", sim.data[ post_spikes_probe ], sim.data[ post_probe ] )
        )
plots.append(
        plotter.plot_weight_matrices_over_time( sim.data[ weight_probe ], sample_every=sample_every )
        )
if n_neurons < 5:
    plots.append(
            plotter.plot_weights_over_time( sim.data[ pos_memr_probe ], sim.data[ neg_memr_probe ] )
            )
    plots.append(
            plotter.plot_values_over_time( 1 / sim.data[ pos_memr_probe ], 1 / sim.data[ neg_memr_probe ] )
            )

dir_name, dir_images = make_timestamped_dir( root="../data/mPES/" )
save_weights( dir_name, sim.data[ weight_probe ] )
for i, fig in enumerate( plots ):
    fig.savefig( dir_images + str( i ) + ".eps", format="eps" )
    fig.show()
