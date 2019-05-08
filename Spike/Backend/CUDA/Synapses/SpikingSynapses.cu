// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikingSynapses);

namespace Backend {
  namespace CUDA {

    __device__ injection_kernel spiking_device_kernel = spiking_current_injection_kernel;
    __device__ synaptic_activation_kernel spiking_syn_activation_kernel = get_active_synapses;

    SpikingSynapses::SpikingSynapses() {
    }

    SpikingSynapses::~SpikingSynapses() {
      for (int u=0; u < synaptic_data->num_synapse_groups; u++){
        CudaSafeCall(cudaFree(h_efferent_synapse_counts[u]));
        CudaSafeCall(cudaFree(h_efferent_synapse_starts[u]));
      }
      CudaSafeCall(cudaFree(efferent_synapse_counts));
      CudaSafeCall(cudaFree(efferent_synapse_starts));
      free(h_efferent_synapse_counts);
      free(h_efferent_synapse_starts);

      CudaSafeCall(cudaFree(delays));
      CudaSafeCall(cudaFree(neuron_inputs.circular_input_buffer));
      CudaSafeCall(cudaFree(d_synaptic_data));
      CudaSafeCall(cudaFree(max_efferents_per_group));
      CudaSafeCall(cudaFree(d_pre_neurons_data));
      CudaSafeCall(cudaFree(d_post_neurons_data));
    }

    void SpikingSynapses::reset_state() {
      Synapses::reset_state();

      // Spike Buffer Resetting
      CudaSafeCall(cudaMemset(circular_input_buffer, 0.0f, sizeof(float)*neuron_inputs.temporal_buffersize*frontend()->pre_neuron_set.size()*frontend()->post_neuron_set[0]->total_number_of_neurons));
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
      neuron_inputs.temporal_buffersize = buffersize;
      neuron_inputs.input_buffersize = frontend()->pre_neuron_set.size()*frontend()->post_neuron_set[0]->total_number_of_neurons;
      neuron_inputs.total_buffersize = neuron_inputs.temporal_buffersize*neuron_inputs.input_buffersize;

      synaptic_data = new spiking_synapses_data_struct();
      synaptic_data->synapse_type = EMPTY;
      synaptic_data->num_synapse_groups = frontend()->post_neuron_set.size();
      
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();
      
      synaptic_data->synapse_neuron_group_indices = synapse_neuron_group_indices;
      synaptic_data->neuron_inputs = neuron_inputs;
      synaptic_data->max_efferents_per_group = max_efferents_per_group;
      synaptic_data->efferent_synapse_counts = efferent_synapse_counts;
      synaptic_data->efferent_synapse_starts = efferent_synapse_starts;
      synaptic_data->postsynaptic_neuron_indices = postsynaptic_neuron_indices;
      synaptic_data->delays = delays;
      synaptic_data->synaptic_efficacies_or_weights = synaptic_efficacies_or_weights;
      synaptic_data->weight_scaling_constants = weight_scaling_constants;


      CudaSafeCall(cudaMemcpy(d_synaptic_data,
        synaptic_data,
        sizeof(spiking_synapses_data_struct),
        cudaMemcpyHostToDevice));
         
    }

    void SpikingSynapses::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&delays, sizeof(int)*frontend()->total_number_of_synapses));
      // Device pointers for spike buffer and active synapse/neuron storage
      CudaSafeCall(cudaMalloc((void **)&d_synaptic_data, sizeof(spiking_synapses_data_struct)));
      // Setting injection kernel
      CudaSafeCall(cudaMemcpyFromSymbol(
        &host_injection_kernel,
        spiking_device_kernel,
        sizeof(injection_kernel)));
      // Setting injection kernel
      CudaSafeCall(cudaMemcpyFromSymbol(
        &host_syn_activation_kernel,
        spiking_syn_activation_kernel,
        sizeof(synaptic_activation_kernel)));

      CudaSafeCall(cudaMalloc((void **)&max_efferents_per_group, sizeof(int)*frontend()->post_neuron_set.size()));
      
      CudaSafeCall(cudaMalloc((void **)&neuron_inputs.circular_input_buffer, sizeof(float*)*frontend()->pre_neuron_set.size()*frontend()->post_neuron_set[0].total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&d_pre_neurons_data, sizeof(spiking_neurons_data_struct*)*frontend()->pre_neuron_set.size()));
      CudaSafeCall(cudaMalloc((void **)&d_post_neurons_data, sizeof(spiking_neurons_data_struct*)*frontend()->post_neuron_set.size()));

      h_efferent_synapse_counts = (int**)malloc(frontend()->efferent_num_per_group.size()*sizeof(int*));
      for (int u=0; u < frontend()->efferent_num_per_group.size(); u++)
        CudaSafeCall(cudaMalloc((void **)&h_efferent_synapse_counts[u], sizeof(int)*frontend()->efferent_num_per_group[u].size()));
      CudaSafeCall(cudaMalloc((void **)&efferent_synapse_counts, sizeof(int*)*frontend()->efferent_num_per_group.size()));

      h_efferent_synapse_starts = (int**)malloc(frontend()->efferent_starts_per_group.size()*sizeof(int*));
      for (int u=0; u < frontend()->efferent_starts_per_group.size(); u++)
        CudaSafeCall(cudaMalloc((void **)&h_efferent_synapse_starts[u], sizeof(int)*frontend()->efferent_starts_per_group[u].size()));
      CudaSafeCall(cudaMalloc((void **)&efferent_synapse_starts, sizeof(int*)*frontend()->efferent_starts_per_group.size()));

    }

    void SpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(delays, frontend()->delays,
        sizeof(int)*frontend()->total_number_of_synapses,
        cudaMemcpyHostToDevice));
      
      
      CudaSafeCall(cudaMemcpy(
        max_efferents_per_group,
        frontend()->maximum_number_of_efferent_synapses_per_group.data(),
        frontend()->maximum_number_of_efferent_synapses_per_group.size()*sizeof(int), cudaMemcpyHostToDevice));
      
      
      for (int u=0; u < frontend()->efferent_num_per_group.size(); u++){
        CudaSafeCall(cudaMemcpy(
              h_efferent_synapse_counts[u],
              frontend()->efferent_num_per_group[u].data(),
              sizeof(int)*frontend()->efferent_num_per_group[u].size(),
              cudaMemcpyHostToDevice));
      }
      CudaSafeCall(cudaMemcpy(
            efferent_synapse_counts,
            h_efferent_synapse_counts,
            sizeof(int*)*frontend()->efferent_num_per_group.size(),
            cudaMemcpyHostToDevice));
      for (int u=0; u < frontend()->efferent_starts_per_group.size(); u++){
        CudaSafeCall(cudaMemcpy(
              h_efferent_synapse_starts[u],
              frontend()->efferent_starts_per_group[u].data(),
              sizeof(int)*frontend()->efferent_starts_per_group[u].size(),
              cudaMemcpyHostToDevice));
      }
      CudaSafeCall(cudaMemcpy(
            efferent_synapse_starts,
            h_efferent_synapse_starts,
            sizeof(int*)*frontend()->efferent_starts_per_group.size(),
            cudaMemcpyHostToDevice));

      for (int u=0; u < frontend()->pre_neuron_set.size(); u++){
        ::Backend::CUDA::SpikingNeurons* neurons_backend =
                  dynamic_cast<::Backend::CUDA::SpikingNeurons*>(frontend()->pre_neuron_set[u]->backend());
        h_pre_neurons_data.push_back(neurons_backend->d_neuron_data);
      }
      CudaSafeCall(cudaMemcpy(d_pre_neurons_data, h_pre_neurons_data.data(),
        sizeof(spiking_neurons_data_struct*)*frontend()->pre_neuron_set.size(),
        cudaMemcpyHostToDevice));

      for (int u=0; u < frontend()->post_neuron_set.size(); u++)
        h_post_neurons_data.push_back((dynamic_cast<::Backend::CUDA::SpikingNeurons*>(frontend()->post_neuron_set[u]->backend()))->d_neuron_data);
      CudaSafeCall(cudaMemcpy(d_post_neurons_data, h_post_neurons_data.data(),
        sizeof(spiking_neurons_data_struct*)*frontend()->post_neuron_set.size(),
        cudaMemcpyHostToDevice));

    }

    void SpikingSynapses::state_update
    (unsigned int current_time_in_timesteps, float timestep) {

      if (frontend()->total_number_of_synapses > 0){
      
      // Calculate buffer location
      int bufferloc = current_time_in_timesteps % buffersize;
      activate_synapses<<<(dynamic_cast<::Backend::CUDA::SpikingNeurons*>(frontend()->post_neuron_set[0]->backend()))->number_of_neuron_blocks_per_grid, threads_per_block>>>(
          d_synaptic_data,
          d_pre_neurons_data,
          d_post_neurons_data,
          bufferloc,
          timestep,
          current_time_in_timesteps,
          frontend()->model->timestep_grouping);
      CudaCheckError();
      }
    }
      

    __global__ void activate_synapses(
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct** pre_neurons_data,
        spiking_neurons_data_struct** post_neurons_data,
        int bufferloc,
        float timestep,
        unsigned int current_time_in_timesteps,
        int timestep_grouping)
    {
      for (int synapse_group = 0; synapse_group < synaptic_data->num_synapse_groups; synapse_group++){
        int indx = threadIdx.x + blockIdx.x * blockDim.x;
        while (indx < (synaptic_data->max_efferents_per_group[synapse_group]*pre_neurons_data[synapse_group]->num_activated_neurons[((current_time_in_timesteps / timestep_grouping) % 2)])) {
          int pos = indx / synaptic_data->max_efferents_per_group[synapse_group];
          int pre_idx = pre_neurons_data[synapse_group]->activated_neuron_ids[pos];
          int idx = indx % synaptic_data->max_efferents_per_group[synapse_group];

          int synapse_count = synaptic_data->efferent_synapse_counts[synapse_group][pre_idx];

          if (idx >= synapse_count){
            indx += blockDim.x * gridDim.x;
            continue;
          }

          int timestep_grouping_index = pre_neurons_data[synapse_group]->activation_timestep_groupings[pos];
          int synapse_id = synaptic_data->efferent_synapse_starts[synapse_group][pre_idx] + idx;
          int postneuron = synaptic_data->postsynaptic_neuron_indices[synapse_id];
          
          int targetloc = (bufferloc + synaptic_data->delays[synapse_id] + timestep_grouping_index) % synaptic_data->neuron_inputs.temporal_buffersize;
          float weightinput = synaptic_data->synaptic_efficacies_or_weights[synapse_id]*synaptic_data->weight_scaling_constants[synapse_id];
          atomicAdd((int*)&(synaptic_data->neuron_inputs.circular_input_buffer[synapsegroup*synaptic_data->neuron_inputs.total_buffersize + targetloc*synaptic_data->neuron_inputs.input_buffersize + postneuron]), weightinput);
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
    
    __device__ void get_active_synapses(
      spiking_synapses_data_struct* synaptic_data,
      spiking_neurons_data_struct* neuron_data,
      int timestep_group_index,
      int preneuron_idx,
      int timestep_index,
      bool is_input) {};

  }
}
