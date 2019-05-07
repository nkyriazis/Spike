#include "ConductanceSpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

// ConductanceSpikingSynapses Destructor
ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {

#ifdef CRAZY_DEBUG
  std::cout << "@@@@@@@@@@ 0 " << synaptic_conductances_g << " \n";
#endif

  free(synaptic_conductances_g);

#ifdef CRAZY_DEBUG
  std::cout << "@@@@@@@@@@ 1\n";
#endif
}


int ConductanceSpikingSynapses::AddGroup(int presynaptic_group_id, 
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

  conductance_spiking_synapse_parameters_struct * conductance_spiking_synapse_group_params = (conductance_spiking_synapse_parameters_struct*)synapse_params;
  
  // Incrementing number of synapses
  ConductanceSpikingSynapses::increment_number_of_synapses(temp_number_of_synapses_in_last_group);

  for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++) {
    synaptic_conductances_g[i] = 0.0f;
  }

  // Set constants
  reversal_potentials_Vhat.push_back(conductance_spiking_synapse_group_params->reversal_potential_Vhat);
  decay_terms_tau_g.push_back(conductance_spiking_synapse_group_params->decay_term_tau_g);
  
  return(groupID);
}

void ConductanceSpikingSynapses::increment_number_of_synapses(int increment) {

  synaptic_conductances_g = (float*)realloc(synaptic_conductances_g, total_number_of_synapses * sizeof(float));
}


void ConductanceSpikingSynapses::state_update(unsigned int current_time_in_timesteps, float timestep) {
  backend()->state_update(current_time_in_timesteps, timestep);
}

SPIKE_MAKE_INIT_BACKEND(ConductanceSpikingSynapses);
