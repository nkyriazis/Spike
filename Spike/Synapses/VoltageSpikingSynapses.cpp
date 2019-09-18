#include "VoltageSpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

// VoltageSpikingSynapses Destructor
VoltageSpikingSynapses::~VoltageSpikingSynapses() {
}


int VoltageSpikingSynapses::AddGroup(int presynaptic_group_id, 
                                          int postsynaptic_group_id, 
                                          Neurons * neurons,
                                          Neurons * input_neurons,
                                          float timestep,
                                          synapse_parameters_struct * synapse_params) {
	
	
  int groupID = SpikingSynapses::AddGroup(presynaptic_group_id, 
                            postsynaptic_group_id, 
                            input_neurons,
                            neurons,
                            timestep,
                            synapse_params);

  voltage_spiking_synapse_parameters_struct * voltage_spiking_synapse_group_params = (voltage_spiking_synapse_parameters_struct*)synapse_params;
  
  // Incrementing number of synapses
  VoltageSpikingSynapses::increment_number_of_synapses(temp_number_of_synapses_in_last_group);

  return(groupID);
}

void VoltageSpikingSynapses::increment_number_of_synapses(int increment) {
}

void VoltageSpikingSynapses::state_update(unsigned int current_time_in_timesteps, float timestep) {
  backend()->state_update(current_time_in_timesteps, timestep);
}

SPIKE_MAKE_INIT_BACKEND(VoltageSpikingSynapses);
