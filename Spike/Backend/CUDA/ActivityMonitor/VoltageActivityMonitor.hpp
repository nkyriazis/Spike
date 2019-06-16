#pragma once

#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.hpp"

#include "Spike/ActivityMonitor/VoltageActivityMonitor.hpp"
#include "ActivityMonitor.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class VoltageActivityMonitor :
      public virtual ::Backend::CUDA::ActivityMonitor,
      public virtual ::Backend::VoltageActivityMonitor {
    public:
      ~VoltageActivityMonitor() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(VoltageActivityMonitor);
      using ::Backend::VoltageActivityMonitor::frontend;

      int max_num_measurements = 1000;
      int num_measurements = 0;
      
      void prepare() override;
      void reset_state() override;

      void allocate_pointers_for_data();

      void copy_data_to_host() override;
      void collect_measurement(unsigned int current_time_in_timesteps, float timestep) override;
    
    private:
      ::SpikingNeurons* neurons_frontend = nullptr;
      ::Backend::CUDA::LIFSpikingNeurons* neurons_backend = nullptr;

    };

    __global__ void add_spikes_to_per_neuron_spike_count_kernel
    (spiking_neurons_data_struct* neuron_data,
     int* d_per_neuron_spike_counts,
     unsigned int current_time_in_timesteps,
     int timestep_grouping,
     size_t total_number_of_neurons);
  }
}
