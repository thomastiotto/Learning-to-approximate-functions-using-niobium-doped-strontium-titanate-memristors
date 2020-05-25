import numpy as np

from nengo.builder import Builder, Operator, Signal
from memristor_nengo.memristors import BidirectionalPowerlaw
from nengo.rc import rc


class SimMemristors( Operator ):
    """Set a memristor model output for the given resistance state.

       Implements ``memristors.step_math(output)``.

       Parameters
       ----------
       memristors : MemristorType
           The `.MemristorType`, which defines a ``step_math`` function.
       output : Signal
           The neuron output signal that will be set.
       states : list, optional
            A list of additional memristor state signals set by ``step_math``.
       tag : str, optional
           A label associated with the operator, for debugging purposes.

       Attributes
       ----------
       memristors : MemristorType
           The `.MemristorType`, which defines a ``step_math`` function.
       output : Signal
           The neuron output signal that will be set.
       states : list, optional
            A list of additional memristor state signals set by ``step_math``.
       tag : str or None
           A label associated with the operator, for debugging purposes.

       Notes
       -----
       1. sets ``[output]+ states``
       2. incs ``[]``
       3. reads ``[]``
       4. updates ``[]``
    """
    
    def __init__( self, memristors, output, states=None, tag=None ):
        super().__init__( tag=tag )
        self.memristors = memristors
        
        self.sets = [ output ] + ([ ] if states is None else states)
        self.incs = [ ]
        self.reads = [ ]
        self.updates = [ ]
    
    @property
    def output( self ):
        return self.sets[ 0 ]
    
    @property
    def states( self ):
        return self.sets[ 1: ]
    
    def _descstr( self ):
        return "%s, %s" % (self.memristors, self.output)
    
    def make_step( self, signals, dt, rng ):
        output = signals[ self.output ]
        states = [ signals[ state ] for state in self.states ]
        
        def step_simmemristors():
            self.memristors.step_math( dt, output, *states )
        
        return step_simmemristors


@Builder.register( BidirectionalPowerlaw )
def build_bidirectionalpowerlaw( model, bidirectionalpowerlaw, memristors ):
    """Builds a `.BidirectionalPowerlawMemristor` object into a model.

    In addition to adding a `.SimMemristors` operator, this build function sets up
    signals to track the resistance for each memristor.

    Parameters
    ----------
    model : Model
        The model to build into.
    bidirectionalpowerlaw: BidirectionalPowerlaw
        Memristor type to build.
    memristor : Memristors
        The memristor population object corresponding to the memristor type.

    Notes
    -----
    TBD
    """
    
    model.sig[ memristors ][ "resistance" ] = Signal(
            shape=memristors.size_in, name="%s.resistance" % memristors
            )
    model.add_op(
            SimMemristors(
                    memristors=bidirectionalpowerlaw,
                    output=model.sig[ memristors ][ "out" ],
                    states=[ model.sig[ memristors ][ "resistance" ] ],
                    )
            )
