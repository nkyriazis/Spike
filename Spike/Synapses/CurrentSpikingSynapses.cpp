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
  
  // Set up unique parameter labels
  bool found = false;
  int param_label = 0;
  for (int p = 0; p < decay_terms_tau.size(); p++){
    if (current_spiking_synapse_group_params->decay_term_tau == decay_terms_tau[p]){
      found = true;
      param_label = p;
    }
  }
  if (!found){
    param_label = decay_terms_tau.size();
    // Set constants
    decay_terms_tau.push_back(current_spiking_synapse_group_params->decay_term_tau);
    // Keep number of parameter labels up to date
    number_of_parameter_labels = param_label + 1;
  }

  for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++) {
    parameter_labels[i] = param_label;
  }
  
  return(groupID);
}


void CurrentSpikingSynapses::state_update(unsigned int current_time_in_timesteps, float timestep) {
  backend()->state_update(current_time_in_timesteps, timestep);
}

SPIKE_MAKE_INIT_BACKEND(CurrentSpikingSynapses);

