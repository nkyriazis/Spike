// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikingSynapses);

namespace Backend {
  namespace CUDA {

    __device__ injection_kernel spiking_device_kernel = spiking_current_injection_kernel;

    SpikingSynapses::SpikingSynapses() {
    }

    SpikingSynapses::~SpikingSynapses() {
      for (int u=0; u < synaptic_data->num_presynaptic_pointers; u++){
        CudaSafeCall(cudaFree(h_efferent_synapse_counts[u]));
        CudaSafeCall(cudaFree(h_efferent_synapse_starts[u]));
      } 
      free(h_efferent_synapse_counts);
      free(h_efferent_synapse_starts);
      CudaSafeCall(cudaFree(efferent_synapse_counts));
      CudaSafeCall(cudaFree(efferent_synapse_starts));
      CudaSafeCall(cudaFree(max_efferents_per_set));
      CudaSafeCall(cudaFree(d_pre_neurons_data));
      
      CudaSafeCall(cudaFree(delays));
      CudaSafeCall(cudaFree(d_syn_labels));
      CudaSafeCall(cudaFree(neuron_inputs.circular_input_buffer));
      CudaSafeCall(cudaFree(d_synaptic_data));
    }

    void SpikingSynapses::reset_state() {
      Synapses::reset_state();

      CudaSafeCall(cudaMemset(neuron_inputs.circular_input_buffer, 0.0f, sizeof(float)*neuron_inputs.temporal_buffersize*neuron_inputs.input_buffersize));
    }

    void SpikingSynapses::copy_weights_to_host() {
      CudaSafeCall(cudaMemcpy(frontend()->synaptic_efficacies_or_weights,
        synaptic_efficacies_or_weights,
        sizeof(float)*frontend()->total_number_of_synapses,
        cudaMemcpyDeviceToHost));
    }

    void SpikingSynapses::prepare() {
      Synapses::prepare();
     
      // Extra buffer size for current time and extra to reset before last
      buffersize = frontend()->maximum_axonal_delay_in_timesteps + 2*frontend()->model->timestep_grouping + 1;
      neuron_inputs.input_buffersize = frontend()->postsynaptic_neuron_pointer->total_number_of_neurons*frontend()->num_syn_labels;
      neuron_inputs.temporal_buffersize = buffersize;
      
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();

      synaptic_data = new spiking_synapses_data_struct();
      synaptic_data->synapse_type = EMPTY;
      synaptic_data->syn_labels = d_syn_labels;
      synaptic_data->num_syn_labels = frontend()->num_syn_labels;
      synaptic_data->neuron_inputs = neuron_inputs;
       
      synaptic_data->num_presynaptic_pointers = frontend()->presynaptic_neuron_pointers.size();
      synaptic_data->presynaptic_pointer_indices = presynaptic_pointer_indices;
      synaptic_data->presynaptic_neuron_indices = presynaptic_neuron_indices;
      synaptic_data->postsynaptic_neuron_indices = postsynaptic_neuron_indices;
      synaptic_data->delays = delays;
      synaptic_data->synaptic_efficacies_or_weights = synaptic_efficacies_or_weights;
     
      synaptic_data->max_efferents_per_set = max_efferents_per_set;
      synaptic_data->efferent_synapse_counts = efferent_synapse_counts;
      synaptic_data->efferent_synapse_starts = efferent_synapse_starts;

      CudaSafeCall(cudaMemcpy(d_synaptic_data,
        synaptic_data,
        sizeof(spiking_synapses_data_struct),
        cudaMemcpyHostToDevice));
         
    }

    void SpikingSynapses::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&delays, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&d_syn_labels, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&d_synaptic_data, sizeof(spiking_synapses_data_struct)));
      // Setting injection kernel
      CudaSafeCall(cudaMemcpyFromSymbol(
        &host_injection_kernel,
        spiking_device_kernel,
        sizeof(injection_kernel)));
      
      CudaSafeCall(cudaMalloc((void **)&neuron_inputs.circular_input_buffer, sizeof(float)*neuron_inputs.temporal_buffersize*neuron_inputs.input_buffersize));

      CudaSafeCall(cudaMalloc((void **)&max_efferents_per_set, sizeof(int)*frontend()->presynaptic_neuron_pointers.size()));
      
      CudaSafeCall(cudaMalloc((void **)&d_pre_neurons_data, sizeof(spiking_neurons_data_struct*)*frontend()->presynaptic_neuron_pointers.size()));
      CudaSafeCall(cudaMalloc((void **)&postsynaptic_neuron_data, sizeof(spiking_neurons_data_struct)));

      h_efferent_synapse_counts = (int**)malloc(frontend()->efferent_num_per_set.size()*sizeof(int*));
      for (int u=0; u < frontend()->efferent_num_per_set.size(); u++)
        CudaSafeCall(cudaMalloc((void **)&h_efferent_synapse_counts[u], sizeof(int)*frontend()->efferent_num_per_set[u].size()));
      CudaSafeCall(cudaMalloc((void **)&efferent_synapse_counts, sizeof(int*)*frontend()->efferent_num_per_set.size()));

      h_efferent_synapse_starts = (int**)malloc(frontend()->efferent_starts_per_set.size()*sizeof(int*));
      for (int u=0; u < frontend()->efferent_starts_per_set.size(); u++)
        CudaSafeCall(cudaMalloc((void **)&h_efferent_synapse_starts[u], sizeof(int)*frontend()->efferent_starts_per_set[u].size()));
      CudaSafeCall(cudaMalloc((void **)&efferent_synapse_starts, sizeof(int*)*frontend()->efferent_starts_per_set.size()));

    }

    void SpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(delays, frontend()->delays,
        sizeof(int)*frontend()->total_number_of_synapses,
        cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        d_syn_labels,
        frontend()->syn_labels,
        sizeof(int)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
     CudaSafeCall(cudaMemcpy(
        max_efferents_per_set,
        frontend()->maximum_number_of_efferent_synapses_per_set.data(),
        frontend()->maximum_number_of_efferent_synapses_per_set.size()*sizeof(int), cudaMemcpyHostToDevice));

     for (int u=0; u < frontend()->efferent_num_per_set.size(); u++){
        CudaSafeCall(cudaMemcpy(
              h_efferent_synapse_counts[u],
              frontend()->efferent_num_per_set[u].data(),
              sizeof(int)*frontend()->efferent_num_per_set[u].size(),
              cudaMemcpyHostToDevice));
      }
      CudaSafeCall(cudaMemcpy(
            efferent_synapse_counts,
            h_efferent_synapse_counts,
            sizeof(int*)*frontend()->efferent_num_per_set.size(),
            cudaMemcpyHostToDevice));
      for (int u=0; u < frontend()->efferent_starts_per_set.size(); u++){
        CudaSafeCall(cudaMemcpy(
              h_efferent_synapse_starts[u],
              frontend()->efferent_starts_per_set[u].data(),
              sizeof(int)*frontend()->efferent_starts_per_set[u].size(),
              cudaMemcpyHostToDevice));
      }
      CudaSafeCall(cudaMemcpy(
            efferent_synapse_starts,
            h_efferent_synapse_starts,
            sizeof(int*)*frontend()->efferent_starts_per_set.size(),
            cudaMemcpyHostToDevice));

      for (int u=0; u < frontend()->presynaptic_neuron_pointers.size(); u++){
        ::Backend::CUDA::SpikingNeurons* neurons_backend =
                  dynamic_cast<::Backend::CUDA::SpikingNeurons*>(frontend()->presynaptic_neuron_pointers[u]->backend());
        h_pre_neurons_data.push_back(neurons_backend->d_neuron_data);
      }
      CudaSafeCall(cudaMemcpy(d_pre_neurons_data, h_pre_neurons_data.data(),
        sizeof(spiking_neurons_data_struct*)*frontend()->presynaptic_neuron_pointers.size(),
        cudaMemcpyHostToDevice));

      postsynaptic_neuron_data = (dynamic_cast<::Backend::CUDA::SpikingNeurons*>(frontend()->postsynaptic_neuron_pointer->backend()))->d_neuron_data;
    }

    void SpikingSynapses::state_update
    (unsigned int current_time_in_timesteps, float timestep) {

      if (frontend()->total_number_of_synapses > 0){
        ::Backend::CUDA::SpikingNeurons* neurons_backend =
                  dynamic_cast<::Backend::CUDA::SpikingNeurons*>(frontend()->postsynaptic_neuron_pointer->backend());
      
        // Calculate buffer location
        int bufferloc = current_time_in_timesteps % buffersize;

        activate_synapses<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
            d_synaptic_data,
            postsynaptic_neuron_data,
            d_pre_neurons_data,
            bufferloc,
            timestep,
            current_time_in_timesteps,
            frontend()->model->timestep_grouping);
        CudaCheckError();
      }
      
    }
      
    __global__ void activate_synapses(
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* post_neuron_data,
        spiking_neurons_data_struct** pre_neurons_data,
        int bufferloc,
        float timestep,
        unsigned int current_time_in_timesteps,
        int timestep_grouping)
    {

      for (int p = 0; p < synaptic_data->num_presynaptic_pointers; p++){
        int indx = threadIdx.x + blockIdx.x * blockDim.x;
        while (indx < (synaptic_data->max_efferents_per_set[p]*pre_neurons_data[p]->num_activated_neurons[((current_time_in_timesteps / timestep_grouping) % 2)])) {
          int pos = indx / synaptic_data->max_efferents_per_set[p]; 
          int idx = indx % synaptic_data->max_efferents_per_set[p]; 
          
          int pre_idx = pre_neurons_data[p]->activated_neuron_ids[pos];
          int synapse_count = synaptic_data->efferent_synapse_counts[p][pre_idx];

          if (idx >= synapse_count){
            indx += blockDim.x * gridDim.x;
            continue;
          }

          int timestep_grouping_index = pre_neurons_data[p]->activation_subtimesteps[pos];
          int synapse_id = synaptic_data->efferent_synapse_starts[p][pre_idx] + idx;
          int postneuron = synaptic_data->postsynaptic_neuron_indices[synapse_id];
          
          int targetloc = (bufferloc + synaptic_data->delays[synapse_id] + timestep_grouping_index) % synaptic_data->neuron_inputs.temporal_buffersize;
          int syn_label = synaptic_data->syn_labels[synapse_id];
          float weightinput = synaptic_data->synaptic_efficacies_or_weights[synapse_id];
          atomicAdd(&(synaptic_data->neuron_inputs.circular_input_buffer[targetloc*synaptic_data->neuron_inputs.input_buffersize + syn_label + postneuron*synaptic_data->num_syn_labels]), weightinput);
          indx += blockDim.x * gridDim.x;
        }
      }
    }

    __device__ float spiking_current_injection_kernel(
        spiking_synapses_data_struct* in_synaptic_data,
        spiking_neurons_data_struct* neuron_data,
        float multiplication_to_volts,
        float current_membrane_voltage,
        unsigned int current_time_in_timesteps,
        float timestep,
        int idx,
        int g){
         return 0.0f;
    }

  }
}
