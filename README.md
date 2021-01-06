# Learning to Approximate Functions Using Nb-doped SrTiO_3 Memristors

## Paper
[arXiv:2011.02794 [cs.ET]](https://arxiv.org/abs/2011.02794)

## Abstract

Memristors have attracted interest as neuromorphic computation elements because they show promise in enabling efficient hardware implementations of artificial neurons and synapses. We performed measurements on interface-type memristors to validate their use in neuromorphic hardware. 

Specifically, we utilised Nb-doped SrTiO3 memristors as synapses in a simulated neural network by arranging them into differential synaptic pairs, with the weight of the connection given by the difference in normalised conductance values between the two paired memristors. This network learned to represent functions through a training process based on a novel supervised learning algorithm, during which discrete voltage pulses were applied to one of the two memristors in each pair. 

To simulate the fact that both the initial state of the physical memristive devices and the impact of each voltage pulse are unknown we injected noise into the simulation. Nevertheless, discrete updates based on local knowledge were shown to result in robust learning performance. 

Using this class of memristive devices as the synaptic weight element in a spiking neural network yields, to our knowledge, one of the first models of this kind, capable of learning to be a universal function approximator, and strongly suggests the suitability of these memristors for usage in future computing platforms.

## Folders
* ``experiments``: various executables used to explore the properties of the memristors
* ``memristor_nengo``: library containing the learning algorithms running in Nengo Core and NengoDL backends, together with extra useful functions
* ``tests``: simple tests for specific functionalities

## Running the code
* ``mPES.py`` runs mPES learning using the simulated memristors and the ``memristor_nengo`` library
* ``averaging_mPES.py`` runs mPES on randomly initialised models and calculates their learning performance statistics
* ``parameter_search_mPES`` runs mPES varying the specified parameter in a chosen range and calculates the learning performance statistics for each parameter value
