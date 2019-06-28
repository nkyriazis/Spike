#pragma once

#include "Spike/Plasticity/InhibitorySTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"

#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class InhibitorySTDPPlasticity : public virtual ::Backend::CUDA::STDPPlasticity,
                           public virtual ::Backend::InhibitorySTDPPlasticity {
    public:
      float* vogels_memory_trace_reset = nullptr; 
      float* vogels_pre_memory_trace = nullptr;
      float* vogels_post_memory_trace = nullptr;
      float* vogels_prevupdate= nullptr;

      ~InhibitorySTDPPlasticity() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(InhibitorySTDPPlasticity);
      using ::Backend::InhibitorySTDPPlasticity::frontend;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers();
      void apply_stdp_to_synapse_weights(unsigned int current_time_in_timesteps, float timestep) override;
    };

    // Kernel to carry out LTP/LTD
    __global__ void vogels_apply_stdp_to_synapse_weights_kernel
          (spiking_synapses_data_struct* synaptic_data,
           spiking_neurons_data_struct* post_neuron_data,
           spiking_neurons_data_struct** pre_neurons_data,
           float* vogels_pre_memory_trace,
           float* vogels_post_memory_trace,
           float trace_decay,
           struct inhibitory_stdp_plasticity_parameters_struct stdp_vars,
           float timestep,
           int timestep_grouping,
           unsigned int current_time_in_timesteps,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses);
  }
}
