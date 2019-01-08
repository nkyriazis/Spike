#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "InputSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"
#include "Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.hpp"
#include "Spike/Backend/CUDA/Synapses/CurrentSpikingSynapses.hpp"
#include "Spike/Backend/CUDA/Synapses/VoltageSpikingSynapses.hpp"

#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Spike/Backend/CUDA/Helpers/RandomStateManager.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class PoissonInputSpikingNeurons : public virtual ::Backend::CUDA::InputSpikingNeurons,
                                       public virtual ::Backend::PoissonInputSpikingNeurons {
    public:
      PoissonInputSpikingNeurons() = default;
      ~PoissonInputSpikingNeurons() override;

      SPIKE_MAKE_BACKEND_CONSTRUCTOR(PoissonInputSpikingNeurons);
      using ::Backend::PoissonInputSpikingNeurons::frontend;

      void prepare() override;
      void reset_state() override;
      void setup_stimulus() override;

      ::Backend::CUDA::RandomStateManager* random_state_manager_backend = nullptr;
      int * next_spike_timestep_of_each_neuron = nullptr;
      float * rates = nullptr;
      bool * active = nullptr;
      bool * init = nullptr;
      
      void allocate_device_pointers(); // Not virtual
      void copy_constants_to_device(); // Not virtual

      void state_update(unsigned int current_time_in_timesteps, float timestep) override;
    };
    __global__ void poisson_update_membrane_potentials_kernel(
        synaptic_activation_kernel syn_activation_kernel,
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* in_neuron_data,
        curandState_t* d_states,
       float *d_rates,
       bool *active,
       float *d_membrane_potentials_v,
       float timestep,
       int timestep_grouping,
       float * d_thresholds_for_action_potential_spikes,
       float* d_resting_potentials,
       int* next_spike_timestep_of_each_neuron,
       unsigned int current_time_in_timesteps,
       size_t total_number_of_input_neurons,
       int current_stimulus_index);
  }
}
