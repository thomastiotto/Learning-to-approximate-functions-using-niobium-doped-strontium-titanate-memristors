import numpy as np


class WeightModifier():
    def __call__( self, weights ):
        return weights


class ZeroShiftModifier( WeightModifier ):
    def __init__( self ):
        super().__init__()
    
    def __call__( self, weights ):
        weights -= np.mean( weights )
        
        return weights


class MemristorLearningRule:
    def __init__( self, learning_rate ):
        self.learning_rate = learning_rate
        self.dt = None
        
        self.rule_name = None
        self.weight_modifier = None
        
        self.input_size = None
        self.output_size = None
        
        self.weights = None
        self.memristors = None
        self.logging = None
        
        self.has_learning_signal = False
        self.has_error_signal = False
        
        # only used in supervised rules (ex. mPES)
        self.last_error = None
    
    def get_error_signal( self ):
        if self.has_error_signal:
            return self.last_error
        else:
            raise ValueError( f"{self.rule_name} takes no error signal" )
    
    def find_spikes( self, input_activities, output_activities=None ):
        spiked_pre = np.tile(
                np.array( np.rint( input_activities ), dtype=bool ), (self.output_size, 1)
                )
        spiked_post = np.tile(
                np.expand_dims(
                        np.array( np.rint( output_activities ), dtype=bool ), axis=1 ), (1, self.input_size)
                ) \
            if output_activities is not None \
            else np.ones( (1, self.input_size) )
        
        return np.logical_and( spiked_pre, spiked_post )


class mHopfieldHebbian( MemristorLearningRule ):
    def __init__( self, learning_rate=1e-6, dt=0.001 ):
        super().__init__( learning_rate, dt )
        
        self.rule_name = "mHopfieldHebbian"
        
        self.alpha = self.learning_rate * self.dt
        self.has_learning_signal = True
    
    def __call__( self, t, x ):
        input_activities = x
        
        spiked_map = self.find_spikes( input_activities, input_activities )
        
        if spiked_map.any():
            for j, i in np.transpose( np.where( spiked_map ) ):
                # ignore diagonal
                if i != j:
                    self.weights[ j, i ] = self.memristors[ j, i ].pulse( spiked_map[ j, i ] )
                    # symmetric update
                    # could also route memristor [j,i] to weight [i,j] like in the paper
                    self.weights[ i, j ] = self.memristors[ i, j ].pulse( spiked_map[ i, j ] )
        # set diagonal to zero
        np.fill_diagonal( self.weights, 0. )
        
        # calculate the output at this timestep
        return np.dot( self.weights, input_activities )


class mOja( MemristorLearningRule ):
    def __init__( self, learning_rate=1e-6, dt=0.001, beta=1.0 ):
        super().__init__( learning_rate, dt )
        
        self.rule_name = "mOja"
        
        self.alpha = self.learning_rate * self.dt
        self.beta = beta
    
    def __call__( self, t, x ):
        input_activities = x[ :self.input_size ]
        output_activities = x[ self.input_size:self.input_size + self.output_size ]
        
        post_squared = self.alpha * output_activities * output_activities
        forgetting = -self.beta * self.weights * np.expand_dims( post_squared, axis=1 )
        hebbian = np.outer( self.alpha * output_activities, input_activities )
        update_direction = hebbian - forgetting
        
        # squash spikes to False (0) or True (100/1000 ...) or everything is always adjusted
        spiked_map = self.find_spikes( input_activities, output_activities )
        
        # we only need to update the weights for the neurons that spiked so we filter
        if spiked_map.any():
            for j, i in np.transpose( np.where( spiked_map ) ):
                self.weights[ j, i ] = self.memristors[ j, i ].pulse( update_direction[ j, i ] )
        
        # calculate the output at this timestep
        return np.dot( self.weights, input_activities )


class mBCM( MemristorLearningRule ):
    def __init__( self, learning_rate=1e-9, dt=0.001 ):
        super().__init__( learning_rate, dt )
        
        self.rule_name = "mBCM"
        
        self.alpha = self.learning_rate * self.dt
    
    def __call__( self, t, x ):
        
        input_activities = x[ :self.input_size ]
        output_activities = x[ self.input_size:self.input_size + self.output_size ]
        theta = x[ self.input_size + self.output_size: ]
        
        update_direction = output_activities - theta
        # function \phi( exponent, \theta ) that is the moving threshold
        update = self.alpha * output_activities * update_direction
        
        # squash spikes to False (0) or True (100/1000 ...) or everything is always adjusted
        spiked_map = self.find_spikes( input_activities, output_activities )
        
        # we only need to update the weights for the neurons that spiked so we filter
        if spiked_map.any():
            for j, i in np.transpose( np.where( spiked_map ) ):
                self.weights[ j, i ] = self.memristors[ j, i ].pulse( update_direction[ j ] )
        
        # calculate the output at this timestep
        return np.dot( self.weights, input_activities )


class mPES( MemristorLearningRule ):
    def __init__( self, encoders, learning_rate=1e-5, error_threshold=1e-6 ):
        super().__init__( learning_rate )
        
        self.rule_name = "mPES"
        
        self.encoders = encoders
        self.error_threshold = error_threshold
        
        self.has_learning_signal = True
        self.has_error_signal = True
    
    def __call__( self, t, x ):
        input_activities = x[ :self.input_size ]
        error = x[ self.input_size: ]
        # squash error to zero under exponent certain threshold (maybe leads to better learning?)
        error[ np.abs( error ) < self.error_threshold ] = 0
        # error = x[ self.input_size: ] if np.abs( x[ self.input_size: ] ) > self.error_threshold else [ 0 ]
        # error = x[ self.input_size: ]
        # note the negative sign, for exponent positive error we want to decrement the output
        alpha = -self.learning_rate * self.dt / self.input_size
        
        self.last_error = error
        
        # we are adjusting weights so calculate local error
        local_error = alpha * np.dot( self.encoders, error )
        
        # include input activities in error calculation
        signal = np.outer( local_error, input_activities )
        
        # squash spikes to False (0) or True (100/1000 ...) or everything is always adjusted
        spiked_map = self.find_spikes( input_activities )
        
        # for j in range( signal.shape[ 0 ] ):
        #     for i in range( signal.shape[ 1 ] ):
        #         if signal[ j, i ] != 0:
        #             update = self.memristors[ j, i ].pulse( signal[ j, i ] )
        #
        #             self.weights[ j, i ] = update
        
        # we only need to update the weights for the neurons that spiked so we filter for their columns
        if spiked_map.any():
            for j, i in np.transpose( np.where( spiked_map ) ):
                update = self.memristors[ j, i ].pulse( signal[ j, i ] )
                # update = update if update >=
                self.weights[ j, i ] = update
        # select each column and pass it to modifier class
        # for i in np.unique( np.transpose( np.where( spiked_map ) )[ :, 1 ] ):
        #     self.weights[ :, i ] = self.weight_modifier( self.weights[ :, i ] )
        
        # calculate the output at this timestep
        return np.dot( self.weights, input_activities )
