#pragma once

#include "Spike/Plasticity/WeightDependentSTDPPlasticity.hpp"
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
    class WeightDependentSTDPPlasticity : public virtual ::Backend::CUDA::STDPPlasticity,
                           public virtual ::Backend::WeightDependentSTDPPlasticity {
    public:
      float* stdp_pre_memory_trace = nullptr;
      float* stdp_post_memory_trace = nullptr;
      float* h_stdp_trace = nullptr;
      
      ~WeightDependentSTDPPlasticity() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(WeightDependentSTDPPlasticity);
      using ::Backend::WeightDependentSTDPPlasticity::frontend;

      void prepare() override;
      void reset_state() override;

      void allocate_device_pointers();
      void apply_stdp_to_synapse_weights(unsigned int current_time_in_timesteps, float timestep) override;
    };
    __global__ void ltp_and_ltd
        (spiking_synapses_data_struct* synaptic_data,
           spiking_neurons_data_struct* post_neuron_data,
           spiking_neurons_data_struct** pre_neurons_data,
           float* stdp_pre_memory_trace,
           float* stdp_post_memory_trace,
           float post_decay,
           float pre_decay,
           weightdependent_stdp_plasticity_parameters_struct stdp_vars,
           float timestep,
           int timestep_grouping,
           int current_time_in_timesteps,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses);

  }
}
