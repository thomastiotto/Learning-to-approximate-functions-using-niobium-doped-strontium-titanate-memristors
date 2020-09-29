# Learning to approximate functions using niobium doped strontium titanate memristors

## Abstract

The ﬁeld of Machine Learning is - at its core - concerned with building function approximators from incomplete data samples. The state of the art approach to solving this problem is in using artiﬁcial neural networks, where a large number of real-valued artiﬁcial neurons are connected to each other by means of weights.
The optimisation process is performed by learning rules that guide the update of the single weights, based on either global (as in the classic back-propagation algorithm) or local knowledge (which is more biologically plausible); the outcome is an interpolation for the hidden mapping from input samples to observed outputs.
Traditional artiﬁcial neural networks are not necessarily constrained by power consumption or other physical limitations and, even though many important advances in deep learning have been “biologically inspired” (e.g., convolutional neural networks), it is unclear how far the current deterministic approach can progress; present-day learning models suﬀer from evident “brittleness” and do not seem well positioned to be able to match human adaptability across a wide variety of tasks. 

Alternative approaches to traditional neural networks aim to implement learning in specialised hardware based on neuromorphic materials. For example, a neuromorphic approach to building a neural network might substitute the continuous neurons in the artiﬁcial neural network for their spiking equivalent and the ideal connection weights for a set of memristors. Memristors are a novel fundamental two-terminal circuit element whose resistance value depends both on the past state of the device and on the input current; because this change in resistance resembles the potentiation and depression of synapses in the brain, there is strong interest in exploring the use of memristors as synapses in neuromorphic circuits. As a consequence, these models need to deal with some of the same constraints that the brain operates under. In particular, neuromorphic learning algorithms can only update weights on the basis of local knowledge and are restricted by the device physics. 

Here, we have utilised niobium doped strontium titanate memristors, whose resistance values in response to voltage pulses follow a power law, as synapses in a simulated neural network.
The model was built using the Nengo Brain Builder framework and consists of pre- and postsynaptic neuronal ensembles, arranged in a fully-connected topology. Each artificial synapse is composed of a pair of memristors, one designated as “positive” and one as “negative”, and the weight of the connection is given by the difference in conductance value between the two paired memristors.
Through a training process - based on adapting existing supervised and unsupervised learning algorithms - where discrete voltage pulses are applied to one of the two memristors in each pair, this model is capable of learning to represent multidimensional functions.
The initial state of the memristive devices is unknown, as are their exact operating parameters, but nonetheless robust learning performance is proven using only discrete updates based on local knowledge.
We demonstrate this principle by instantiating increasingly larger pre- and post-synaptic neuronal ensembles and applying our learning rules to modulate the memristor-based connections. 
 The experiments simulate 30 seconds of neuronal activity divided into 22 seconds of learning (“exploration”) and the remaining of testing the weights (“exploitation”). In separate experiments, the pre-synaptic neuronal ensemble is fed either a band-limited white noise signal or a set of sine waves uniformly phase-shifted across dimensions.
This process yields a post-synaptic ensemble capable of correctly representing the original input signal, however high-dimensional it may be, provided that the neuronal ensembles are large enough to have sufficient representational power. We also show that we can learn transformations of the original signal across the synaptic connection, which is a harder problem than learning a communication channel.

## Folders
* ``experiments``: various executables used to explore the properties of the memristors
* ``memristor_learning``: library containing the learning functions running in a Nengo Node object *[deprecated]*
* ``memristor_nengo``: library containing the learning functions running in Nengo Core backend
* ``tests``: simple tests for specific functionalities

## Running the code
* ``mPES.py`` runs mPES learning using the simulated memristors and the ``memristor_nengo`` library
* ``averaging_mPES.py`` runs mPES on randomly initialised models and calculates their learning performance statistics
* ``parameter_search_mPES`` runs mPES varying the specified parameter in a chosen range and calculates the learning performance statistics for each parameter value
