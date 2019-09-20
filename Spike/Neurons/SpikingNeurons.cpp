#include "SpikingNeurons.hpp"
#include <stdlib.h>

// SpikingNeurons Constructor
SpikingNeurons::SpikingNeurons() {
}

// SpikingNeurons Destructor
SpikingNeurons::~SpikingNeurons() {
}

void SpikingNeurons::prepare_backend_early() {
}

int SpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
  int new_group_id = Neurons::AddGroup(group_params);

  spiking_neuron_parameters_struct * spiking_group_params = (spiking_neuron_parameters_struct*)group_params;


  return new_group_id;
}

void SpikingNeurons::state_update(unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping) {
  backend()->state_update(current_time_in_timesteps, timestep, timestep_grouping);
}

SPIKE_MAKE_INIT_BACKEND(SpikingNeurons);
