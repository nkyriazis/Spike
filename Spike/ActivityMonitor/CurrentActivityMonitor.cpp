#include "CurrentActivityMonitor.hpp"
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <string>
#include <time.h>
using namespace std;

// CurrentActivityMonitor Constructor
CurrentActivityMonitor::CurrentActivityMonitor(CurrentSpikingSynapses * synapses_point, int id, int label){
  synapses = synapses_point;
  neuron_id = id;
  label_id = label;
}

void CurrentActivityMonitor::reset_state() {
  free(measurements);
  measurements = nullptr;
  num_measurements = 0;
  backend()->reset_state();
}

void CurrentActivityMonitor::state_update(unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping) {
  backend()->collect_measurement(current_time_in_timesteps, timestep, timestep_grouping);
}

void CurrentActivityMonitor::final_update(unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping){
  backend()->copy_data_to_host();
}

void CurrentActivityMonitor::save_measurements_as_txt(string path, string prefix){
  ofstream measurementsfile;

  // Open output files
  measurementsfile.open((path + "/" + prefix + "Conductances.txt"), ios::out | ios::binary);

  // Send the data
  for (int i = 0; i < num_measurements; i++) {
    measurementsfile << measurements[i] << endl;
  }
  // Close the files
  measurementsfile.close();
}

void CurrentActivityMonitor::save_measurements_as_binary(string path, string prefix){
  ofstream measurementsfile;

  // Open output files
  measurementsfile.open((path + "/" + prefix + "Currents.bin"), ios::out | ios::binary);

  // Send the data
  measurementsfile.write((char *)measurements, num_measurements*sizeof(float));
  // Close the files
  measurementsfile.close();
}

SPIKE_MAKE_INIT_BACKEND(CurrentActivityMonitor);
