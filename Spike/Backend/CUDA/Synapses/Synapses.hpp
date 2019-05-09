#pragma once

#include "Spike/Synapses/Synapses.hpp"

#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Helpers/RandomStateManager.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    struct synapses_data_struct{
    };

    class Synapses : public virtual ::Backend::Synapses {
    public:
      ~Synapses() override;
      using ::Backend::Synapses::frontend;

      int* presynaptic_neuron_indices = nullptr;
      int* postsynaptic_neuron_indices = nullptr;
      int* temp_presynaptic_neuron_indices = nullptr;
      int* temp_postsynaptic_neuron_indices = nullptr;
      float* synaptic_efficacies_or_weights = nullptr;
      float* temp_synaptic_efficacies_or_weights = nullptr;
      float * weight_scaling_constants = nullptr;
      int * synapse_set_indices = nullptr;
      int max_num_blocks_per_grid = 0;
      
      // CUDA Specific
      ::Backend::CUDA::RandomStateManager* random_state_manager_backend = nullptr;
      dim3 number_of_synapse_blocks_per_grid;
      dim3 threads_per_block;
      synapses_data_struct* synaptic_data;
      synapses_data_struct* d_synaptic_data;

      void prepare() override;
      void reset_state() override;
      void copy_to_frontend() override;
      void copy_to_backend() override;

      void allocate_device_pointers(); // Not virtual
      void copy_constants_and_initial_efficacies_to_device(); // Not virtual
      void set_threads_per_block_and_blocks_per_grid(int threads); // Not virtual

    };
  }
}
