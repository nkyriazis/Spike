// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/ActivityMonitor/ConductanceActivityMonitor.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, ConductanceActivityMonitor);

namespace Backend {
  namespace CUDA {
    ConductanceActivityMonitor::~ConductanceActivityMonitor() {
    }

    void ConductanceActivityMonitor::reset_state() {
      ActivityMonitor::reset_state();
      num_measurements = 0;
    }

    void ConductanceActivityMonitor::prepare() {
      ActivityMonitor::prepare();
      synapses_frontend = frontend()->synapses;
      synapses_backend =
        dynamic_cast<::Backend::CUDA::ConductanceSpikingSynapses*>(synapses_frontend->backend());
    }

    void ConductanceActivityMonitor::copy_data_to_host(){
      frontend()->measurements = (float*)realloc(frontend()->measurements, sizeof(float)*(frontend()->num_measurements + num_measurements));
      for (int i = 0; i < num_measurements; i++){
        frontend()->measurements[frontend()->num_measurements + i] = measurements[i];
      }
      frontend()->num_measurements += num_measurements;
      reset_state();
    }

    void ConductanceActivityMonitor::collect_measurement
    (unsigned int current_time_in_timesteps, float timestep) {
      CudaSafeCall(cudaMemcpy(frontend()->measurements + num_measurements,
                              synapses_backend->synaptic_data->neuron_wise_conductance_trace + (frontend()->label_id + frontend()->neuron_id*synapses_backend->synaptic_data->num_syn_labels),
                              sizeof(float), 
                              cudaMemcpyDeviceToHost));

      num_measurements++;

      if (num_measurements == max_num_measurements)
        copy_data_to_host();
    }


  }
}

