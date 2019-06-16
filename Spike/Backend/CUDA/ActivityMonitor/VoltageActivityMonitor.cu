// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/ActivityMonitor/VoltageActivityMonitor.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, VoltageActivityMonitor);

namespace Backend {
  namespace CUDA {
    VoltageActivityMonitor::~VoltageActivityMonitor() {
    free(measurements);
    }

    void VoltageActivityMonitor::reset_state() {
      ActivityMonitor::reset_state();
      num_measurements = 0;
    }

    void VoltageActivityMonitor::prepare() {
      ActivityMonitor::prepare();
      neurons_frontend = frontend()->neurons;
      neurons_backend =
        dynamic_cast<::Backend::CUDA::LIFSpikingNeurons*>(neurons_frontend->backend());
      measurements = (float*)realloc(measurements, sizeof(float)*(max_num_measurements));
    }

    void VoltageActivityMonitor::copy_data_to_host(){
      frontend()->neuron_measurements = (float*)realloc(frontend()->neuron_measurements, sizeof(float)*(frontend()->num_measurements + num_measurements));
      for (int i = 0; i < num_measurements; i++){
        frontend()->neuron_measurements[frontend()->num_measurements + i] = measurements[i];
      }
      frontend()->num_measurements += num_measurements;
      reset_state();
    }

    void VoltageActivityMonitor::collect_measurement
    (unsigned int current_time_in_timesteps, float timestep) {
      CudaSafeCall(cudaMemcpy(measurements + num_measurements,
                              neurons_backend->membrane_potentials_v + frontend()->neuron_id,
                              sizeof(float), 
                              cudaMemcpyDeviceToHost));

      num_measurements++;

      if (num_measurements == max_num_measurements)
        copy_data_to_host();
    }


  }
}

