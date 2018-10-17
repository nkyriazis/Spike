//  Inhibitory STDPPlasticity Class C++
//  InhibitorySTDPPlasticity.cu
//


#include "InhibitorySTDPPlasticity.hpp"
#include "../Helpers/TerminalHelpers.hpp"

InhibitorySTDPPlasticity::InhibitorySTDPPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_plasticity_parameters_struct* stdp_parameters){
  stdp_params = (inhibitory_stdp_plasticity_parameters_struct *)stdp_parameters;
}

InhibitorySTDPPlasticity::~InhibitorySTDPPlasticity() {
}

void InhibitorySTDPPlasticity::prepare_backend_late() {
}

// Run the STDP
void InhibitorySTDPPlasticity::state_update(int current_time_in_timesteps, float timestep){
  backend()->apply_stdp_to_synapse_weights(current_time_in_timesteps, timestep);
}

SPIKE_MAKE_INIT_BACKEND(InhibitorySTDPPlasticity);
