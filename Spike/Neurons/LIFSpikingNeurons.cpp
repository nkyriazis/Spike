#include "LIFSpikingNeurons.hpp"
#include <stdlib.h>
#include <stdio.h>


LIFSpikingNeurons::LIFSpikingNeurons() {
}


LIFSpikingNeurons::~LIFSpikingNeurons() {
}

void LIFSpikingNeurons::prepare_backend_early() {
  SpikingNeurons::prepare_backend_early();
  assert(backend() && "Backend needs to have been prepared before calling this!");
  if (!random_state_manager) {
    random_state_manager = new RandomStateManager();
    random_state_manager->init_backend(backend()->context);
  }
}


int LIFSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){

  int new_group_id = SpikingNeurons::AddGroup(group_params);

  lif_spiking_neuron_parameters_struct * this_group_params = (lif_spiking_neuron_parameters_struct*)group_params;

  bool found = false;
  int index = 0;
  for (int label = 0; label <  after_spike_reset_potentials_vreset.size(); label++){
    int matches = 0;
    if (this_group_params->after_spike_reset_potential_vreset == after_spike_reset_potentials_vreset[label])
      matches += 1;
    if (this_group_params->resting_potential_v0 == resting_potentials_v0[label])
      matches += 1;
    if (this_group_params->threshold_for_action_potential_spike == spiking_thresholds_vthresh[label])
      matches += 1;
    float tmem = this_group_params->somatic_capacitance_Cm / this_group_params->somatic_leakage_conductance_g0;
    if (tmem == membrane_time_constants_tau_m[label])
      matches += 1;
    if ((1.0 / this_group_params->somatic_leakage_conductance_g0) == membrane_resistances_R[label])
      matches += 1;
    if (this_group_params->background_current == background_currents[label])
      matches += 1;
    if (this_group_params->absolute_refractory_period == refractory_periods[label])
      matches += 1;

    if ((this_group_params->adaptation == adaptations[label]) && (this_group_params->adaptation_reversal_potential == adaptation_reversal_potentials[label]) && (this_group_params->adaptation_strength == adaptation_strengths[label]) && (this_group_params->adaptation_tau == adaptation_taus[label])){
      matches += 1;
    } else if ((!this_group_params->adaptation) && (!adaptations[label])){
      matches += 1;
    }

    if (matches == 8){
      index = label;
      found = true;
      break;
    }
  }

  if (!found) {
    after_spike_reset_potentials_vreset.push_back(this_group_params->after_spike_reset_potential_vreset);
    resting_potentials_v0.push_back(this_group_params->resting_potential_v0);
    spiking_thresholds_vthresh.push_back(this_group_params->threshold_for_action_potential_spike);
    membrane_time_constants_tau_m.push_back(this_group_params->somatic_capacitance_Cm / this_group_params->somatic_leakage_conductance_g0);
    membrane_resistances_R.push_back(1.0 / this_group_params->somatic_leakage_conductance_g0);
    background_currents.push_back(this_group_params->background_current);
    refractory_periods.push_back(this_group_params->absolute_refractory_period);

    adaptations.push_back(this_group_params->adaptation);
    adaptation_reversal_potentials.push_back(this_group_params->adaptation_reversal_potential);
    adaptation_strengths.push_back(this_group_params->adaptation_strength);
    adaptation_taus.push_back(this_group_params->adaptation_tau);
    
    index = after_spike_reset_potentials_vreset.size() - 1;
  }

  for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
    neuron_labels.push_back(index);
    if (!this_group_params->set_init_membrane){
      membrane_potentials_v.push_back(this_group_params->resting_potential_v0);
    } else {
      membrane_potentials_v.push_back(this_group_params->membrane_potential_range[0] + ((float)(rand()) / RAND_MAX)*(this_group_params->membrane_potential_range[1] - this_group_params->membrane_potential_range[0]));
    }
  }


  return new_group_id;
}

SPIKE_MAKE_INIT_BACKEND(LIFSpikingNeurons);

