#pragma once

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include "Neurons.hpp"

#include <cuda.h>
#include <vector_types.h>

//#define BITLOC(current_time_in_seconds, timestep, offset, bufsizebytes) (((int)ceil(current_time_in_seconds / timestep) + offset) % bufsizebytes*8)
//(((int)(ceil(current_time_in_seconds / timestep)) + g) % (8*bufsizebytes))
#define BYTELOC(bitloc) (bitloc / 8)
#define SUBBITLOC(bitloc) (bitloc % 8)

namespace Backend {
  namespace CUDA {
    struct spiking_neurons_data_struct : neurons_data_struct {
        int* num_activated_neurons;
        int* activated_neuron_ids;
        int* activation_subtimesteps;

        uint8_t* neuron_spike_time_bitbuffer;
        int* neuron_spike_time_bitbuffer_bytesize;
    };

    class SpikingNeurons : public virtual ::Backend::CUDA::Neurons,
                           public virtual ::Backend::SpikingNeurons {
    public:
      SpikingNeurons();
      ~SpikingNeurons() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(SpikingNeurons);
      using ::Backend::SpikingNeurons::frontend;

      void prepare() override;
      void reset_state() override;

      // Device Pointers
      int* num_activated_neurons = nullptr;
      int* activated_neuron_ids = nullptr;
      int* activation_subtimesteps = nullptr;

      // Keeping neuorn spike times
      int h_neuron_spike_time_bitbuffer_bytesize;
      int* neuron_spike_time_bitbuffer_bytesize = nullptr;
      uint8_t* neuron_spike_time_bitbuffer = nullptr;

      spiking_neurons_data_struct* neuron_data;
      spiking_neurons_data_struct* d_neuron_data;

      /**  
       *  Exclusively for the allocation of device memory. This class requires allocation of d_current_injections only.
      */
      void allocate_device_pointers(); // Not virtual

      /**  
       *  Allows copying of static data related to neuron dynamics to the device.
       */
      void copy_constants_to_device(); // Not virtual

      void state_update(unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping) override;
      
    };

  }
} // namespace Backend
