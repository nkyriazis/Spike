#include "CurrentSpikingSynapses.hpp"
#include "Spike/Helpers/TerminalHelpers.hpp"

int CurrentSpikingSynapses::AddGroup(int presynaptic_group_id, 
                                      int postsynaptic_group_id, 
                                      Neurons * neurons,
                                      Neurons * input_neurons,
                                      float timestep,
                                      synapse_parameters_struct * synapse_params) {
  
  int groupID = SpikingSynapses::AddGroup(presynaptic_group_id, 
                            postsynaptic_group_id, 
                            neurons,
                            input_neurons,
                            timestep,
                            synapse_params);

  current_spiking_synapse_parameters_struct * current_spiking_synapse_group_params = (current_spiking_synapse_parameters_struct*)synapse_params;
  
  decay_terms_tau.push_back(current_spiking_synapse_group_params->decay_term_tau);
  return(groupID);
}


void CurrentSpikingSynapses::state_update(unsigned int current_time_in_timesteps, float timestep) {
  backend()->state_update(current_time_in_timesteps, timestep);
}

SPIKE_MAKE_INIT_BACKEND(CurrentSpikingSynapses);

