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
                  neurons=4,
                  base_voltage=1e-1,
                  simulation_time=30.0,
                  learning_time=15.0,
                  simulation_step=0.001,
                  function_to_learn=lambda x: x,
                  seed=None,
                  weights_to_plot=None ):
        self.memristor_controller = memristor_controller
        self.memristor_model = memristor_model
        self.learning_rule = learning_rule
        self.voltage_converter = voltage_converter
        
        self.neurons = neurons
        self.base_voltage = base_voltage
        self.simulation_time = simulation_time
        self.learning_time = learning_time
        self.simulation_step = simulation_step
        self.function_to_learn = function_to_learn
        self.seed = seed
        
        self.input_period = 4
        self.input_frequency = 1 / self.input_period
        self.pre_nrn = neurons
        self.post_nrn = neurons
        self.err_nrn = neurons
        
        self.weights_to_plot = range( 0, int( self.learning_time + 1 ), 1 ) if weights_to_plot is None \
            else weights_to_plot
        
        print( f"Neurons: {neurons}" )
        print( f"Base voltage: {base_voltage}" )
        print( f"Simulation time: {simulation_time}" )
        print( f"Leaning time: {learning_time}" )
        print( f"Simulation step: {simulation_step}" )
        print( f"Function to learn: {function_to_learn}" )
        print( f"Seed: {seed}" )
    
    def __call__( self, ):
        with nengo.Network() as model:
            inp = nengo.Node(
                    output=lambda t: np.sin( self.input_frequency * 2 * np.pi * t ),
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
                    base_voltage=self.base_voltage
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
            
            # plot_ensemble( sim, inp_probe )
            # plot_ensemble_spikes( sim, "Pre", pre_spikes_probe, pre_probe )
            # plot_ensemble_spikes( sim, "Post", post_spikes_probe, post_probe )
            plot_pre_post( sim, pre_probe, post_probe, inp_probe, memr_arr.get_history( "error" ),
                           time=self.learning_time )
            if self.neurons <= 10:
                stats = memr_arr.get_stats( time=(0, self.learning_time), select="conductance" )
                memr_arr.plot_state( sim,
                                     "conductance",
                                     combined=True,
                                     figsize=(15, 10),
                                     # ylim=(0, stats[ "max" ])
                                     # ylim=(0, 2.2e-8)
                                     # upper limit found by looking at the max obtained with memristor pair
                                     )
            for t in self.weights_to_plot:
                memr_arr.plot_weight_matrix( time=t )
            
            print()
            print( "Mean squared error:", mse( sim, inp_probe, post_probe, self.learning_time, self.simulation_step ) )
            print( f"Starting sparsity: {sparsity_measure( memr_arr.get_history( 'weight' )[ 0 ] )}" )
            print( f"Ending sparsity: {sparsity_measure( memr_arr.get_history( 'weight' )[ -1 ] )}" )
            print()
