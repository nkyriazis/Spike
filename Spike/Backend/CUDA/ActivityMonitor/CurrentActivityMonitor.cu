// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/ActivityMonitor/CurrentActivityMonitor.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, CurrentActivityMonitor);

namespace Backend {
  namespace CUDA {
    CurrentActivityMonitor::~CurrentActivityMonitor() {
    }

    void CurrentActivityMonitor::reset_state() {
      ActivityMonitor::reset_state();
      num_measurements = 0;
    }

    void CurrentActivityMonitor::prepare() {
      ActivityMonitor::prepare();
      synapses_frontend = frontend()->synapses;
      synapses_backend =
        dynamic_cast<::Backend::CUDA::CurrentSpikingSynapses*>(synapses_frontend->backend());
      measurements = (float*)realloc(measurements, sizeof(float)*(max_num_measurements));
    }

    void CurrentActivityMonitor::copy_data_to_host(){
      frontend()->measurements = (float*)realloc(frontend()->measurements, sizeof(float)*(frontend()->num_measurements + num_measurements));
      for (int i = 0; i < num_measurements; i++){
        frontend()->measurements[frontend()->num_measurements + i] = measurements[i];
      }
      frontend()->num_measurements += num_measurements;
      reset_state();
    }

    void CurrentActivityMonitor::collect_measurement
    (unsigned int current_time_in_timesteps, float timestep) {
      CudaSafeCall(cudaMemcpy(measurements + num_measurements,
                              synapses_backend->neuron_wise_current_trace + (frontend()->label_id + frontend()->neuron_id*synapses_frontend->num_syn_labels),
                              sizeof(float), 
                              cudaMemcpyDeviceToHost));

      num_measurements++;

      if (num_measurements == max_num_measurements)
        copy_data_to_host();
    }


  }
}

