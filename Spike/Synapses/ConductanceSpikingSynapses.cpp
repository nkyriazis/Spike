#include "ConductanceSpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

// ConductanceSpikingSynapses Destructor
ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {
}


int ConductanceSpikingSynapses::AddGroup(int presynaptic_group_id, 
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

  conductance_spiking_synapse_parameters_struct * conductance_spiking_synapse_group_params = (conductance_spiking_synapse_parameters_struct*)synapse_params;
  
  // Incrementing number of synapses
  ConductanceSpikingSynapses::increment_number_of_synapses(temp_number_of_synapses_in_last_group);

  if (reversal_potentials_Vhat.size() == 0){
    // If a group has not yet been initialized, make it of this type
    reversal_potentials_Vhat.push_back(conductance_spiking_synapse_group_params->reversal_potential_Vhat);
    decay_terms_tau_g.push_back(conductance_spiking_synapse_group_params->decay_term_tau_g);
    weight_scaling_constants.push_back(conductance_spiking_synapse_group_params->weight_scaling_constant);
  } else {
    // Check if this pair exists, if yes set the syn_labels or create a new syn_label
    bool isfound = false;
    int indextoset = 0;
    for (int index = 0; index < reversal_potentials_Vhat.size(); index++){
      if (  (reversal_potentials_Vhat[index] == conductance_spiking_synapse_group_params->reversal_potential_Vhat) &&
        (decay_terms_tau_g[index] == conductance_spiking_synapse_group_params->decay_term_tau_g) &&
        (weight_scaling_constants[index] == conductance_spiking_synapse_group_params->weight_scaling_constant)){
        isfound = true;
        indextoset = index;
        break;
      }
    }
    if (!isfound){
      reversal_potentials_Vhat.push_back(conductance_spiking_synapse_group_params->reversal_potential_Vhat);
      decay_terms_tau_g.push_back(conductance_spiking_synapse_group_params->decay_term_tau_g);
      weight_scaling_constants.push_back(conductance_spiking_synapse_group_params->weight_scaling_constant);
      indextoset = num_syn_labels;
      num_syn_labels++;

    }
    // Now set the synapse labels
    for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++) {
      syn_labels[i] = indextoset;
    }
  }

  return(groupID);
}

void ConductanceSpikingSynapses::increment_number_of_synapses(int increment) {
}


void ConductanceSpikingSynapses::state_update(unsigned int current_time_in_timesteps, float timestep) {
  backend()->state_update(current_time_in_timesteps, timestep);
}

SPIKE_MAKE_INIT_BACKEND(ConductanceSpikingSynapses);
