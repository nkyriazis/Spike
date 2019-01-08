//  CustomSTDPPlasticity STDP Class C++
//  CustomSTDPPlasticity.cu
//
//  Author: Nasir Ahmad
//  Date: 03/10/2016


#include "CustomSTDPPlasticity.hpp"
#include "../Helpers/TerminalHelpers.hpp"

CustomSTDPPlasticity::CustomSTDPPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_plasticity_parameters_struct* stdp_parameters){
  stdp_params = (custom_stdp_plasticity_parameters_struct *)stdp_parameters;
}

CustomSTDPPlasticity::~CustomSTDPPlasticity() {
}

void CustomSTDPPlasticity::prepare_backend_late() {
}


// Run the STDP
void CustomSTDPPlasticity::state_update(unsigned int current_time_in_timesteps, float timestep){
  backend()->apply_stdp_to_synapse_weights(current_time_in_timesteps, timestep);
}


SPIKE_MAKE_INIT_BACKEND(CustomSTDPPlasticity);
