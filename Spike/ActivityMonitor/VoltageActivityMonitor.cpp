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
  neuron_id = id;
}

void VoltageActivityMonitor::reset_state() {
  free(neuron_measurements);
  neuron_measurements = nullptr;
  num_measurements = 0;
  backend()->reset_state();
}

void VoltageActivityMonitor::state_update(unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping) {
  backend()->collect_measurement(current_time_in_timesteps, timestep, timestep_grouping);
}

void VoltageActivityMonitor::final_update(unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping){
  backend()->copy_data_to_host();
}

void VoltageActivityMonitor::save_measurements_as_txt(string path, string prefix){
  ofstream measurementsfile;

  // Open output files
  measurementsfile.open((path + "/" + prefix + "MemVoltages.txt"), ios::out | ios::binary);

  // Send the data
  for (int i = 0; i < num_measurements; i++) {
    measurementsfile << neuron_measurements[i] << endl;
  }
  // Close the files
  measurementsfile.close();
}

void VoltageActivityMonitor::save_measurements_as_binary(string path, string prefix){
  ofstream measurementsfile;

  // Open output files
  measurementsfile.open((path + "/" + prefix + "MemVoltages.bin"), ios::out | ios::binary);

  // Send the data
  measurementsfile.write((char *)neuron_measurements, num_measurements*sizeof(float));
  // Close the files
  measurementsfile.close();
}

SPIKE_MAKE_INIT_BACKEND(VoltageActivityMonitor);
