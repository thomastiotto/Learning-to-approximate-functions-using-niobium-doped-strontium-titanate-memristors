from memristor_learning.MemristorHelpers import *
from memristor_learning.MemristorControllers import *
from memristor_learning.MemristorLearningRules import *
from memristor_learning.MemristorModels import *


class SupervisedLearning():
    def __init__( self,
                  memristor_controller,
                  memristor_model,
                  learning_rule=mPES,
                  voltage_converter=VoltageConverter,
                  weight_modifier=WeightModifier,
                  neurons=4,
                  base_voltage=1e-1,
                  gain=1e5,
                  simulation_time=30.0,
                  learning_time=None,
                  simulation_step=0.001,
                  input_function=lambda t: np.sin( 1 / 4 * 2 * np.pi * t ),
                  function_to_learn=lambda x: x,
                  seed=None,
                  weights_to_plot=None,
                  plot_ylim=(0, 2e-8),  # upper limit chosen by looking at the max obtained with mPlusMinus
                  verbose=True,
                  generate_figures=True ):
        
        self.memristor_controller = memristor_controller
        self.memristor_model = memristor_model
        self.learning_rule = learning_rule
        self.voltage_converter = voltage_converter
        self.weight_modifier = weight_modifier
        
        self.neurons = neurons
        self.base_voltage = base_voltage
        self.gain = gain
        self.simulation_time = simulation_time
        self.simulation_step = simulation_step
        self.input_function = input_function
        self.function_to_learn = function_to_learn
        self.seed = seed
        if not learning_time:
            self.learning_time = self.simulation_time / 2
        
        self.plot_ylim = plot_ylim
        self.verbose = verbose
        self.generate_figures = generate_figures
        
        self.pre_nrn = neurons
        self.post_nrn = neurons
        self.err_nrn = 2 * neurons
        
        self.weights_to_plot = range( 0, int( self.learning_time + 1 ), 1 ) if weights_to_plot is None \
            else weights_to_plot
        
        if self.verbose:
            print( f"Neurons: {self.neurons}" )
            print( f"Base voltage: {self.base_voltage}" )
            print( f"Gain: {self.gain}" )
            print( f"Simulation time: {self.simulation_time}" )
            print( f"Learning time: {self.learning_time}" )
            print( f"Simulation step: {self.simulation_step}" )
            print( f"Function to learn: {self.function_to_learn}" )
            print( f"Seed: {self.seed}" )
    
    def __call__( self, ):
        with nengo.Network() as model:
            inp = nengo.Node(
                    output=self.input_function,
                    size_out=1,
                    label="Input"
                    )
            learning_switch = nengo.Node(
                    lambda t, x: 1 if t < self.learning_time else 0,
                    size_in=1
                    )
            
            pre = nengo.Ensemble(
                    n_neurons=self.pre_nrn,
                    dimensions=1,
                    encoders=generate_encoders( self.pre_nrn ),
                    label="Pre",
                    seed=self.seed
                    )
            post = nengo.Ensemble(
                    n_neurons=self.post_nrn,
                    dimensions=1,
                    encoders=generate_encoders( self.post_nrn ),
                    label="Post",
                    seed=self.seed
                    )
            error = nengo.Ensemble(
                    n_neurons=self.err_nrn,
                    dimensions=1,
                    radius=2,
                    label="Error",
                    seed=self.seed
                    )
            
            # TODO get encoders at runtime as sim.data[ens].encoders
            memr_arr = MemristorArray(
                    model=self.memristor_model,
                    learning_rule=self.learning_rule( encoders=post.encoders ),
                    in_size=self.pre_nrn,
                    out_size=self.post_nrn,
                    seed=self.seed,
                    voltage_converter=self.voltage_converter(),
                    weight_modifier=self.weight_modifier(),
                    base_voltage=self.base_voltage,
                    gain=self.gain
                    )
            learn = nengo.Node(
                    output=memr_arr,
                    size_in=self.pre_nrn + error.dimensions + 1,
                    size_out=self.post_nrn,
                    label="Learn"
                    )
            
            nengo.Connection( inp, pre )
            nengo.Connection( pre.neurons, learn[ :self.pre_nrn ], synapse=0.005 )
            nengo.Connection( post, error )
            nengo.Connection( pre, error, function=self.function_to_learn, transform=-1 )
            nengo.Connection( error, learn[ self.pre_nrn:-1 ] )
            nengo.Connection( learn, post.neurons, synapse=None )
            nengo.Connection( learning_switch, learn[ -1 ], synapse=None )
            
            inp_probe = nengo.Probe( inp )
            pre_spikes_probe = nengo.Probe( pre.neurons )
            post_spikes_probe = nengo.Probe( post.neurons )
            pre_probe = nengo.Probe( pre, synapse=0.01 )
            post_probe = nengo.Probe( post, synapse=0.01 )
            
            # plot_network( model )
            
            with nengo.Simulator( model, dt=self.simulation_step ) as sim:
                sim.run( self.simulation_time )
            
            fig_ensemble = None
            fig_pre = None
            fig_post = None
            fig_pre_post = None
            fig_state = None
            figs_weights = None
            figs_weights_norm = None
            figs_conductances = None
            
            stats = memr_arr.get_stats( time=(0, self.learning_time), select="conductance" )
            if self.generate_figures:
                fig_ensemble = plot_ensemble( sim, [ inp_probe ] )
                fig_pre = plot_ensemble_spikes( sim, "Pre", pre_spikes_probe, pre_probe )
                fig_post = plot_ensemble_spikes( sim, "Post", post_spikes_probe, post_probe )
                fig_pre_post = plot_pre_post( sim, pre_probe, post_probe, [ inp_probe ],
                                              memr_arr.get_history( "error" ),
                                              time=self.learning_time )
                if self.neurons <= 10:
                    fig_state = memr_arr.plot_state( sim,
                                                     "conductance",
                                                     combined=True,
                                                     figsize=(15, 10),
                                                     # ylim=(0, stats[ "max" ])
                                                     ylim=self.plot_ylim
                                                     )
                figs_weights = [ ]
                figs_weights_norm = [ ]
                figs_conductances = [ ]
                for t in self.weights_to_plot:
                    figs_weights.append( memr_arr.plot_weight_matrix( time=t ) )
                    figs_weights_norm.append( memr_arr.plot_weight_matrix( time=t, normalized=True ) )
                    figs_conductances.append( memr_arr.plot_conductance_matrix( time=t ) )
            
            mean_squared_error = mse( sim, inp_probe, post_probe, self.learning_time, self.simulation_step )
            start_sparsity = sparsity_measure( memr_arr.get_history( 'weight' )[ 0 ] )
            end_sparsity = sparsity_measure( memr_arr.get_history( 'weight' )[ -1 ] )
            
            if self.verbose:
                print()
                print( f"Mean squared error: {mean_squared_error}" )
                print( f"Starting sparsity: {start_sparsity}" )
                print( f"Ending sparsity: {end_sparsity}" )
                print()
            
            output = {
                    "stats"            : stats,
                    "mse"              : mean_squared_error,
                    "initial_sparsity" : start_sparsity,
                    "end_sparsity"     : end_sparsity,
                    "fig_ensemble"     : fig_ensemble,
                    "fig_pre"          : fig_pre,
                    "fig_post"         : fig_post,
                    "fig_pre_post"     : fig_pre_post,
                    "fig_state"        : fig_state,
                    "figs_weights"     : figs_weights,
                    "figs_weights_norm": figs_weights_norm,
                    "figs_conductances": figs_conductances,
                    }
            
            # plt.close( "all" )
            
            return output


class SupervisedLearning2D():
    def __init__( self,
                  memristor_controller,
                  memristor_model,
                  learning_rule=mPES,
                  voltage_converter=VoltageConverter,
                  weight_modifier=WeightModifier,
                  neurons=4,
                  base_voltage=1e-1,
                  gain=1e5,
                  simulation_time=30.0,
                  learning_time=None,
                  simulation_step=0.001,
                  input_function=lambda t: np.sin( 1 / 4 * 2 * np.pi * t ),
                  function_to_learn=lambda x: x,
                  seed=None,
                  weights_to_plot=None,
                  plot_ylim=(0, 2e-8),  # upper limit chosen by looking at the max obtained with mPlusMinus
                  verbose=True,
                  generate_figures=True ):
        
        self.memristor_controller = memristor_controller
        self.memristor_model = memristor_model
        self.learning_rule = learning_rule
        self.voltage_converter = voltage_converter
        self.weight_modifier = weight_modifier
        
        self.neurons = neurons
        self.base_voltage = base_voltage
        self.gain = gain
        self.simulation_time = simulation_time
        self.simulation_step = simulation_step
        self.input_function = input_function
        self.function_to_learn = function_to_learn
        self.seed = seed
        if not learning_time:
            self.learning_time = self.simulation_time / 2
        
        self.plot_ylim = plot_ylim
        self.verbose = verbose
        self.generate_figures = generate_figures
        
        self.pre_nrn = neurons
        self.post_nrn = neurons
        self.err_nrn = 2 * neurons
        
        self.weights_to_plot = range( 0, int( self.learning_time + 1 ), 1 ) if weights_to_plot is None \
            else weights_to_plot
        
        if self.verbose:
            print( f"Neurons: {self.neurons}" )
            print( f"Base voltage: {self.base_voltage}" )
            print( f"Gain: {self.gain}" )
            print( f"Simulation time: {self.simulation_time}" )
            print( f"Learning time: {self.learning_time}" )
            print( f"Simulation step: {self.simulation_step}" )
            print( f"Function to learn: {self.function_to_learn}" )
            print( f"Seed: {self.seed}" )
    
    def __call__( self, ):
        with nengo.Network() as model:
            inp = nengo.Node(
                    output=lambda t: (np.sin( 1 / 4 * 2 * np.pi * t ), np.sin( 1 / 4 * 2 * np.pi * t + np.pi )),
                    size_out=2,
                    label="Input"
                    )
            learning_switch = nengo.Node(
                    lambda t, x: 1 if t < self.learning_time else 0,
                    size_in=1
                    )
            
            pre = nengo.Ensemble(
                    n_neurons=self.pre_nrn,
                    dimensions=2,
                    encoders=generate_encoders( self.pre_nrn, dimensions=2 ),
                    label="Pre",
                    seed=self.seed
                    )
            post = nengo.Ensemble(
                    n_neurons=self.post_nrn,
                    dimensions=2,
                    encoders=generate_encoders( self.post_nrn, dimensions=2 ),
                    label="Post",
                    seed=self.seed
                    )
            error = nengo.Ensemble(
                    n_neurons=self.err_nrn,
                    dimensions=2,
                    radius=2,
                    label="Error",
                    seed=self.seed
                    )
            
            # TODO get encoders at runtime as sim.data[ens].encoders (?)
            memr_arr = MemristorArray(
                    model=self.memristor_model,
                    learning_rule=self.learning_rule( encoders=post.encoders ),
                    in_size=self.pre_nrn,
                    out_size=self.post_nrn,
                    seed=self.seed,
                    voltage_converter=self.voltage_converter(),
                    weight_modifier=self.weight_modifier(),
                    base_voltage=self.base_voltage,
                    gain=self.gain
                    )
            learn = nengo.Node(
                    output=memr_arr,
                    size_in=self.pre_nrn + error.dimensions + 1,
                    size_out=self.post_nrn,
                    label="Learn"
                    )
            
            nengo.Connection( inp, pre )
            # nengo.Connection( inp_two, pre[ 1 ] )
            nengo.Connection( pre.neurons, learn[ :self.pre_nrn ], synapse=0.005 )
            nengo.Connection( post, error )
            nengo.Connection( pre, error, function=self.function_to_learn, transform=-1 )
            nengo.Connection( error, learn[ self.pre_nrn:-1 ] )
            nengo.Connection( learn, post.neurons, synapse=None )
            nengo.Connection( learning_switch, learn[ -1 ], synapse=None )
            
            inp_probe_one = nengo.Probe( inp )
            pre_spikes_probe = nengo.Probe( pre.neurons )
            post_spikes_probe = nengo.Probe( post.neurons )
            pre_probe = nengo.Probe( pre, synapse=0.01 )
            post_probe = nengo.Probe( post, synapse=0.01 )
            
            # plot_network( model )
            
            with nengo.Simulator( model, dt=self.simulation_step ) as sim:
                sim.run( self.simulation_time )
            
            fig_ensemble = None
            fig_pre = None
            fig_post = None
            fig_pre_post = None
            fig_state = None
            figs_weights = None
            figs_weights_norm = None
            figs_conductances = None
            
            stats = memr_arr.get_stats( time=(0, self.learning_time), select="conductance" )
            if self.generate_figures:
                fig_ensemble = plot_ensemble( sim, [ inp_probe_one, inp_probe_one ] )
                fig_pre = plot_ensemble_spikes( sim, "Pre", pre_spikes_probe, pre_probe )
                fig_post = plot_ensemble_spikes( sim, "Post", post_spikes_probe, post_probe )
                fig_pre_post = plot_pre_post( sim, pre_probe, post_probe, [ inp_probe_one, inp_probe_one ],
                                              memr_arr.get_history( "error" ),
                                              time=self.learning_time )
                if self.neurons <= 10:
                    fig_state = memr_arr.plot_state( sim,
                                                     "conductance",
                                                     combined=True,
                                                     figsize=(15, 10),
                                                     # ylim=(0, stats[ "max" ])
                                                     ylim=self.plot_ylim
                                                     )
                figs_weights = [ ]
                figs_weights_norm = [ ]
                figs_conductances = [ ]
                for t in self.weights_to_plot:
                    figs_weights.append( memr_arr.plot_weight_matrix( time=t ) )
                    figs_weights_norm.append( memr_arr.plot_weight_matrix( time=t, normalized=True ) )
                    figs_conductances.append( memr_arr.plot_conductance_matrix( time=t ) )
            
            # mean_squared_error = mse( sim, inp_probe, post_probe, self.learning_time, self.simulation_step )
            start_sparsity = sparsity_measure( memr_arr.get_history( 'weight' )[ 0 ] )
            end_sparsity = sparsity_measure( memr_arr.get_history( 'weight' )[ -1 ] )
            
            if self.verbose:
                print()
                # print( f"Mean squared error: {mean_squared_error}" )
                print( f"Starting sparsity: {start_sparsity}" )
                print( f"Ending sparsity: {end_sparsity}" )
                print()
            
            output = {
                    "stats"            : stats,
                    # "mse"              : mean_squared_error,
                    "initial_sparsity" : start_sparsity,
                    "end_sparsity"     : end_sparsity,
                    "fig_ensemble"     : fig_ensemble,
                    "fig_pre"          : fig_pre,
                    "fig_post"         : fig_post,
                    "fig_pre_post"     : fig_pre_post,
                    "fig_state"        : fig_state,
                    "figs_weights"     : figs_weights,
                    "figs_weights_norm": figs_weights_norm,
                    "figs_conductances": figs_conductances,
                    }
            
            # plt.close( "all" )
            
            return output
