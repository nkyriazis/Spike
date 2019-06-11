#include "VoltageActivityMonitor.hpp"
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <string>
#include <time.h>
using namespace std;

// VoltageActivityMonitor Constructor
VoltageActivityMonitor::VoltageActivityMonitor(SpikingNeurons * neurons_parameter, int id){
  neurons = neurons_parameter;
  neuron_id = neuron_id;
}

void VoltageActivityMonitor::reset_state() {
  free(neuron_membrane_voltages);
  neuron_membrane_voltages = nullptr;
  num_measurements = 0;
  backend()->reset_state();
}

void VoltageActivityMonitor::prepare_backend_early() {
  per_neuron_spike_counts = (int *)malloc(sizeof(int)*neurons->total_number_of_neurons);
}

void VoltageActivityMonitor::state_update(unsigned int current_time_in_timesteps, float timestep) {
  backend()->collect_measurement(current_time_in_timesteps, timestep);
}

void VoltageActivityMonitor::final_update(unsigned int current_time_in_timesteps, float timestep){
  backend()->copy_data_to_host();
}

SPIKE_MAKE_INIT_BACKEND(RateActivityMonitor);
