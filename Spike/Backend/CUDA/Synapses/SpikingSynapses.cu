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
      CudaSafeCall(cudaFree(delays));
      CudaSafeCall(cudaFree(num_active_synapses));
      CudaSafeCall(cudaFree(num_activated_neurons));
      CudaSafeCall(cudaFree(active_synapse_counts));
      CudaSafeCall(cudaFree(active_synapse_starts));
      CudaSafeCall(cudaFree(active_presynaptic_neuron_indices));
      CudaSafeCall(cudaFree(neuron_inputs.circular_input_buffer));
      CudaSafeCall(cudaFree(d_synaptic_data));
      CudaSafeCall(cudaFree(max_efferents_per_group));
      CudaSafeCall(cudaFree(d_pre_neurons_data));
      CudaSafeCall(cudaFree(d_post_neurons_data));
    }

    void SpikingSynapses::reset_state() {
      Synapses::reset_state();

      // Spike Buffer Resetting
      CudaSafeCall(cudaMemset(num_activated_neurons, 0, 2*sizeof(int)));
      for (int u = 0; u < frontend()->post_neuron_set.size(); u++){
        CudaSafeCall(cudaMemset(h_circular_input_buffer[u], 0.0f, sizeof(float)*neuron_inputs.temporal_buffersize*frontend()->post_neuron_set[u]->total_number_of_neurons));
      }
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
      if (h_input_buffersize)
        free(h_input_buffersize);
      h_input_buffersize = (int*)malloc((int)frontend()->post_neuron_set.size()*sizeof(int));
      h_circular_input_buffer = (float**)malloc((int)frontend()->post_neuron_set.size()*sizeof(float*));
      for (int u=0; u < frontend()->post_neuron_set.size(); u++)
        h_input_buffersize[u] = frontend()->post_neuron_set[u]->total_number_of_neurons;
      neuron_inputs.temporal_buffersize = buffersize;

      
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();
      
      synaptic_data = new spiking_synapses_data_struct();
      synaptic_data->synapse_type = EMPTY;
      synaptic_data->num_synapse_groups = frontend()->post_neuron_set.size();
      synaptic_data->synapse_neuron_group_indices = synapse_neuron_group_indices;
      synaptic_data->neuron_inputs = neuron_inputs;
      synaptic_data->max_efferents_per_group = max_efferents_per_group;
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
      CudaSafeCall(cudaMalloc((void **)&group_indices, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&num_active_synapses, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&num_activated_neurons, 2*sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&active_synapse_counts, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&active_synapse_starts, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&active_presynaptic_neuron_indices, sizeof(int)*(frontend()->total_number_of_synapses)));
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

      CudaSafeCall(cudaMalloc((void **)&neuron_inputs.input_buffersize, sizeof(int)*frontend()->post_neuron_set.size()));
      CudaSafeCall(cudaMalloc((void **)&neuron_inputs.circular_input_buffer, sizeof(float*)*frontend()->post_neuron_set.size()));
      CudaSafeCall(cudaMalloc((void **)&max_efferents_per_group, sizeof(int)*frontend()->post_neuron_set.size()));
      for (int u=0; u < frontend()->post_neuron_set.size(); u++)
        CudaSafeCall(cudaMalloc((void **)&h_circular_input_buffer[u], sizeof(float)*neuron_inputs.temporal_buffersize*h_input_buffersize[u]));
      
      CudaSafeCall(cudaMalloc((void **)&d_pre_neurons_data, sizeof(spiking_neurons_data_struct)*frontend()->pre_neuron_set.size()));
      CudaSafeCall(cudaMalloc((void **)&d_post_neurons_data, sizeof(spiking_neurons_data_struct)*frontend()->post_neuron_set.size()));

    }

    void SpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(delays, frontend()->delays,
        sizeof(int)*frontend()->total_number_of_synapses,
        cudaMemcpyHostToDevice));
      
      
      CudaSafeCall(cudaMemcpy(neuron_inputs.input_buffersize, h_input_buffersize,
        sizeof(int)*frontend()->post_neuron_set.size(),
        cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(neuron_inputs.circular_input_buffer, h_circular_input_buffer,
        sizeof(float*)*frontend()->post_neuron_set.size(),
        cudaMemcpyHostToDevice));
      
      CudaSafeCall(cudaMemcpy(
        max_efferents_per_group,
        frontend()->maximum_number_of_efferent_synapses_per_group.data(),
        frontend()->maximum_number_of_efferent_synapses_per_group.size()*sizeof(int), cudaMemcpyHostToDevice));

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



      activate_synapses<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
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
      
    __device__ void get_active_synapses(
      spiking_synapses_data_struct* synaptic_data,
      spiking_neurons_data_struct* neuron_data,
      int timestep_group_index,
      int preneuron_idx,
      int grouping_index,
      bool is_input)
    {
      int pos = atomicAdd(&synaptic_data->num_activated_neurons[grouping_index % 2], 1);
      int synapse_count = neuron_data->per_neuron_efferent_synapse_count[preneuron_idx];
      int synapse_start = neuron_data->per_neuron_efferent_synapse_start[preneuron_idx];
      synaptic_data->active_synapse_counts[pos] = synapse_count;
      synaptic_data->active_synapse_starts[pos] = synapse_start;
      synaptic_data->group_indices[pos] = timestep_group_index;
    };
      

    __global__ void activate_synapses(
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* pre_neurons_data,
        spiking_neurons_data_struct* post_neurons_data,
        int bufferloc,
        float timestep,
        unsigned int current_time_in_timesteps,
        int timestep_grouping)
    {
      for (int synapse_group = 0; synapse_group < synaptic_data->num_synapse_groups; synapse_group++){
        int indx = threadIdx.x + blockIdx.x * blockDim.x;
        if (indx == 0){
          pre_neurons_data->num_activated_neurons[((current_time_in_timesteps / timestep_grouping) + 1) % 2] = 0;
        }
        while (indx < (synaptic_data->max_efferents_per_group[synapse_group]*pre_neurons_data->num_activated_neurons[((current_time_in_timesteps / timestep_grouping) % 2)])) {
          int pos = indx / synaptic_data->max_efferents_per_group[synapse_group]; 
          int idx = indx % synaptic_data->max_efferents_per_group[synapse_group]; 

          // NEEDS TO CHANGE TO A SYNAPSE VARIABLE BASED UPON THE ACTIVE NEURONS WHICH THE NEURON CLASS GIVES US
          int synapse_count = synaptic_data->active_synapse_counts[pos];

          if (idx >= synapse_count){
            indx += blockDim.x * gridDim.x;
            continue;
          }

          int synapse_id = synaptic_data->active_synapse_starts[pos] + idx;
          int postneuron = synaptic_data->postsynaptic_neuron_indices[synapse_id];
          
          int targetloc = (bufferloc + synaptic_data->delays[synapse_id] + synaptic_data->group_indices[pos]) % synaptic_data->neuron_inputs.temporal_buffersize;
          float weightinput = synaptic_data->synaptic_efficacies_or_weights[synapse_id]*synaptic_data->weight_scaling_constants[synapse_id];
          atomicAdd((int*)&(synaptic_data->neuron_inputs.circular_input_buffer[synapse_group][targetloc*synaptic_data->neuron_inputs.input_buffersize[synapse_group] + postneuron]), weightinput);
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
