#include "ConductanceActivityMonitor.hpp"
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <string>
#include <time.h>
using namespace std;

// ConductanceActivityMonitor Constructor
ConductanceActivityMonitor::ConductanceActivityMonitor(ConductanceSpikingSynapses * synapses_point, int id, int label){
  synapses = synapses_point;
  neuron_id = neuron_id;
  syn_label = syn_label;
}

void ConductanceActivityMonitor::reset_state() {
  free(measurements);
  measurements = nullptr;
  num_measurements = 0;
  backend()->reset_state();
}

void ConductanceActivityMonitor::state_update(unsigned int current_time_in_timesteps, float timestep) {
  backend()->collect_measurement(current_time_in_timesteps, timestep);
}

void ConductanceActivityMonitor::final_update(unsigned int current_time_in_timesteps, float timestep){
  backend()->copy_data_to_host();
}

void ConductanceActivityMonitor::save_measurements_as_txt(string path, string prefix){
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

void ConductanceActivityMonitor::save_measurements_as_binary(string path, string prefix){
  ofstream measurementsfile;

  // Open output files
  measurementsfile.open((path + "/" + prefix + "Conductances.bin"), ios::out | ios::binary);

  // Send the data
  measurementsfile.write((char *)measurements, num_measurements*sizeof(float));
  // Close the files
  measurementsfile.close();
}

SPIKE_MAKE_INIT_BACKEND(ConductanceActivityMonitor);
