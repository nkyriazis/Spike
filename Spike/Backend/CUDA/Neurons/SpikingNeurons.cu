// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikingNeurons);

namespace Backend {
  namespace CUDA {
    SpikingNeurons::SpikingNeurons() {
    }

    SpikingNeurons::~SpikingNeurons() {
      CudaSafeCall(cudaFree(d_neuron_data));
      CudaSafeCall(cudaFree(num_activated_neurons));
      CudaSafeCall(cudaFree(activated_neuron_ids));
      CudaSafeCall(cudaFree(activation_subtimesteps));

      CudaSafeCall(cudaFree(neuron_spike_time_bitbuffer_bytesize));
      CudaSafeCall(cudaFree(neuron_spike_time_bitbuffer));
    }

    void SpikingNeurons::allocate_device_pointers() {
      
      CudaSafeCall(cudaMalloc((void **)&num_activated_neurons, sizeof(int)*2));
      CudaSafeCall(cudaMalloc((void **)&activated_neuron_ids, sizeof(int)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&activation_subtimesteps, sizeof(int)*frontend()->total_number_of_neurons));

      CudaSafeCall(cudaMalloc((void **)&d_neuron_data, sizeof(spiking_neurons_data_struct)));
      
      h_neuron_spike_time_bitbuffer_bytesize = ((frontend()->model->spiking_synapses->maximum_axonal_delay_in_timesteps + frontend()->model->timestep_grouping) / 8) + 1;
      CudaSafeCall(cudaMalloc((void **)&neuron_spike_time_bitbuffer_bytesize, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&neuron_spike_time_bitbuffer, sizeof(uint8_t)*frontend()->total_number_of_neurons*h_neuron_spike_time_bitbuffer_bytesize));
    }

    void SpikingNeurons::copy_constants_to_device() {
    }

    void SpikingNeurons::prepare() {
      Neurons::prepare();
      allocate_device_pointers();
      copy_constants_to_device();

      neuron_data = new spiking_neurons_data_struct();
      memcpy(neuron_data, (static_cast<SpikingNeurons*>(this)->Neurons::neuron_data), sizeof(neurons_data_struct));

      neuron_data->num_activated_neurons = num_activated_neurons;
      neuron_data->activated_neuron_ids = activated_neuron_ids;
      neuron_data->activation_subtimesteps = activation_subtimesteps;
      neuron_data->neuron_spike_time_bitbuffer = neuron_spike_time_bitbuffer;
      neuron_data->neuron_spike_time_bitbuffer_bytesize = neuron_spike_time_bitbuffer_bytesize;


      CudaSafeCall(cudaMemcpy(
        d_neuron_data, 
        neuron_data,
        sizeof(spiking_neurons_data_struct), cudaMemcpyHostToDevice));
    }

    void SpikingNeurons::reset_state() {
      Neurons::reset_state();

      CudaSafeCall(cudaMemset(num_activated_neurons, 0, (sizeof(int)*2)));
      CudaSafeCall(cudaMemset(neuron_spike_time_bitbuffer, 0, (sizeof(uint8_t)*frontend()->total_number_of_neurons*h_neuron_spike_time_bitbuffer_bytesize)));
      CudaSafeCall(cudaMemcpy(neuron_spike_time_bitbuffer_bytesize,
                              &h_neuron_spike_time_bitbuffer_bytesize,
                              sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    
    void SpikingNeurons::state_update(unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping) {
    }


  } // ::Backend::CUDA
} // ::Backend
