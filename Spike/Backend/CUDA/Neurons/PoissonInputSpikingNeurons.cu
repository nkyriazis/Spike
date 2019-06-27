// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Neurons/PoissonInputSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, PoissonInputSpikingNeurons);

namespace Backend {
  namespace CUDA {
    namespace INLINE_POIS {
      #include "Spike/Backend/CUDA/InlineDeviceFunctions.hpp"
    }
    PoissonInputSpikingNeurons::~PoissonInputSpikingNeurons() {
      CudaSafeCall(cudaFree(next_spike_timestep_of_each_neuron));
      CudaSafeCall(cudaFree(rates));
      CudaSafeCall(cudaFree(active));
      if (init)
        free(init);
    }

    void PoissonInputSpikingNeurons::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&next_spike_timestep_of_each_neuron, sizeof(int)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&rates, sizeof(float)*frontend()->total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&active, sizeof(bool)*frontend()->total_number_of_neurons));
      init = (bool*)malloc(sizeof(bool)*frontend()->total_number_of_neurons);
      for (int n=0; n < frontend()->total_number_of_neurons; n++)
        init[n] = false;
      CudaSafeCall(cudaMemcpy(active, init, sizeof(bool)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
    }

    void PoissonInputSpikingNeurons::copy_constants_to_device() {
      if (frontend()->rates) {
        CudaSafeCall(cudaMemcpy(rates, frontend()->rates, sizeof(float)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
      }
    }

    void PoissonInputSpikingNeurons::reset_state() {
      InputSpikingNeurons::reset_state();
      CudaSafeCall(cudaMemcpy(active, init, sizeof(bool)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
    }
    
    void PoissonInputSpikingNeurons::setup_stimulus() {
      CudaSafeCall(cudaMemcpy(active, init, sizeof(bool)*frontend()->total_number_of_neurons, cudaMemcpyHostToDevice));
    }

    void PoissonInputSpikingNeurons::prepare() {
      InputSpikingNeurons::prepare();

      allocate_device_pointers();
      copy_constants_to_device();

      // Crudely assume that the RandomStateManager backend is also CUDA:
      random_state_manager_backend
        = dynamic_cast<::Backend::CUDA::RandomStateManager*>
        (frontend()->random_state_manager->backend());
      assert(random_state_manager_backend);
    }

    void PoissonInputSpikingNeurons::state_update(unsigned int current_time_in_timesteps, float timestep) {
      ::Backend::CUDA::SpikingSynapses* synapses_backend =
        dynamic_cast<::Backend::CUDA::SpikingSynapses*>(frontend()->model->spiking_synapses->backend());
      poisson_update_membrane_potentials_kernel<<<random_state_manager_backend->block_dimensions, random_state_manager_backend->threads_per_block>>>(
         synapses_backend->host_syn_activation_kernel,
         synapses_backend->d_synaptic_data,
         d_neuron_data,
         random_state_manager_backend->states,
         rates,
         active,
         timestep,
         frontend()->model->timestep_grouping,
         next_spike_timestep_of_each_neuron,
         current_time_in_timesteps,
         frontend()->total_number_of_neurons,
         frontend()->current_stimulus_index);

  CudaCheckError();
    }

    __global__ void poisson_update_membrane_potentials_kernel(
        synaptic_activation_kernel syn_activation_kernel,
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* in_neuron_data,
        curandState_t* d_states,
       float *d_rates,
       bool *active,
       float timestep,
       int timestep_grouping,
       int* next_spike_timestep_of_each_neuron,
       unsigned int current_time_in_timesteps,
       size_t total_number_of_input_neurons,
       int current_stimulus_index) {

   
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx == 0){
        in_neuron_data->num_activated_neurons[((current_time_in_timesteps / timestep_grouping) + 1) % 2] = 0;
      }
      
      int bufsize = in_neuron_data->neuron_spike_time_bitbuffer_bytesize[0];

      while (idx < total_number_of_input_neurons){
            
        for (int g=0; g < timestep_grouping; g++){
            int bitloc = (current_time_in_timesteps + g) % (8*bufsize);
            // Creates random float between 0 and 1 from uniform distribution
            // d_states effectively provides a different seed for each thread
            // curand_uniform produces different float every time you call it
            in_neuron_data->neuron_spike_time_bitbuffer[idx*bufsize + (bitloc / 8)] &= ~(1 << (bitloc % 8));
            if ((next_spike_timestep_of_each_neuron[idx] <= 0) || (!active[idx])){
                //(next_spike_time_of_each_neuron[idx] <= ((current_time_in_timesteps + g)*timestep)) || (!active[idx])){
              int rate_index = (total_number_of_input_neurons * current_stimulus_index) + idx;
              float rate = d_rates[rate_index];
              float random_float = curand_uniform(&d_states[t_idx]);
              next_spike_timestep_of_each_neuron[idx] = (int)ceilf((- (1.0f / rate)*logf(random_float))/timestep);//(current_time_in_timesteps + g)*timestep +  - (1.0f / rate)*logf(random_float);
              if (active[idx]){
                in_neuron_data->neuron_spike_time_bitbuffer[idx*bufsize + (bitloc / 8)] |= (1 << (bitloc % 8));
                // Recording the neuron which has spiked and which sub-timestep within which it did
                int pos = atomicAdd(&in_neuron_data->num_activated_neurons[(current_time_in_timesteps / timestep_grouping) % 2], 1);
                in_neuron_data->activated_neuron_ids[pos] = idx;
                in_neuron_data->activation_subtimesteps[pos] = g;
              } else {
                active[idx] = true;
              }
            } else {
              next_spike_timestep_of_each_neuron[idx] -= 1;
            }
        } 
      idx += blockDim.x * gridDim.x;
      }
    }

  }
}
