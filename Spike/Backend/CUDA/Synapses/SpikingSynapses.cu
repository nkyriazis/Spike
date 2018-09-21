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
      CudaSafeCall(cudaFree(d_syn_labels));
      CudaSafeCall(cudaFree(group_indices));
      CudaSafeCall(cudaFree(num_active_synapses));
      CudaSafeCall(cudaFree(num_activated_neurons));
      CudaSafeCall(cudaFree(active_synapse_counts));
      CudaSafeCall(cudaFree(active_presynaptic_neuron_indices));
      CudaSafeCall(cudaFree(neuron_inputs.circular_input_buffer));
      CudaSafeCall(cudaFree(d_synaptic_data));
    }

    void SpikingSynapses::reset_state() {
      Synapses::reset_state();

      // Spike Buffer Resetting
      CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(num_activated_neurons, 0, sizeof(int)));
      CudaSafeCall(cudaMemset(neuron_inputs.circular_input_buffer, 0.0f, sizeof(float)*neuron_inputs.temporal_buffersize*neuron_inputs.input_buffersize));
      CudaSafeCall(cudaMemset(neuron_inputs.bufferloc, 0, sizeof(int)));
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
      neuron_inputs.input_buffersize = frontend()->neuron_pop_size*frontend()->num_syn_labels;
      neuron_inputs.temporal_buffersize = buffersize;
      
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();

      synaptic_data = new spiking_synapses_data_struct();
      synaptic_data->synapse_type = EMPTY;
      synaptic_data->num_syn_labels = frontend()->num_syn_labels;
      synaptic_data->neuron_inputs = neuron_inputs;
      synaptic_data->num_activated_neurons = num_activated_neurons;
      synaptic_data->num_active_synapses = num_active_synapses;
      synaptic_data->active_presynaptic_neuron_indices = active_presynaptic_neuron_indices;
      synaptic_data->active_synapse_counts = active_synapse_counts;
      synaptic_data->group_indices = group_indices;

      CudaSafeCall(cudaMemcpy(d_synaptic_data,
        synaptic_data,
        sizeof(spiking_synapses_data_struct),
        cudaMemcpyHostToDevice));
         
    }

    void SpikingSynapses::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&delays, sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&d_syn_labels, sizeof(int)*frontend()->total_number_of_synapses));
      // Device pointers for spike buffer and active synapse/neuron storage
      CudaSafeCall(cudaMalloc((void **)&group_indices, sizeof(int)*(frontend()->total_number_of_synapses)));
      CudaSafeCall(cudaMalloc((void **)&num_active_synapses, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&num_activated_neurons, sizeof(int)));
      CudaSafeCall(cudaMalloc((void **)&active_synapse_counts, sizeof(int)*(frontend()->total_number_of_synapses)));
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

      CudaSafeCall(cudaMalloc((void **)&neuron_inputs.circular_input_buffer, sizeof(float)*neuron_inputs.temporal_buffersize*neuron_inputs.input_buffersize));
      CudaSafeCall(cudaMalloc((void **)&neuron_inputs.bufferloc, sizeof(int)));
    }

    void SpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(delays, frontend()->delays,
        sizeof(int)*frontend()->total_number_of_synapses,
        cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        d_syn_labels,
        frontend()->syn_labels,
        sizeof(int)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
    }

    void SpikingSynapses::state_update
    (::SpikingNeurons* neurons,
     ::SpikingNeurons* input_neurons,
     float current_time_in_seconds, float timestep) {

      if (frontend()->total_number_of_synapses > 0){
      
      // Calculate buffer location
      int bufferloc = (int)(std::round(current_time_in_seconds / timestep)) % buffersize;
      //synaptic_data->neuron_inputs = neuron_inputs;


      ::Backend::CUDA::SpikingNeurons* neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(neurons->backend());
      assert(neurons_backend);
      ::Backend::CUDA::SpikingNeurons* input_neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(input_neurons->backend());
      assert(input_neurons_backend);

      /*
      CudaSafeCall(cudaMemcpy(
          &h_num_active_synapses,
          num_active_synapses,
          sizeof(int), cudaMemcpyDeviceToHost));
      int blocks_per_grid = ((h_num_active_synapses / threads_per_block.x) + 1);
      if (blocks_per_grid > max_num_blocks_per_grid) blocks_per_grid = max_num_blocks_per_grid;
      */
      //activate_synapses<<<blocks_per_grid, threads_per_block>>>(
      activate_synapses<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(
        input_neurons_backend->d_neuron_data,
        neurons_backend->d_neuron_data,
        bufferloc,
        buffersize,
        synaptic_data->neuron_inputs,
        postsynaptic_neuron_indices,
        presynaptic_neuron_indices,
        synaptic_efficacies_or_weights,
        weight_scaling_constants,
        delays,
        frontend()->num_syn_labels,
        d_syn_labels,
        timestep,
        current_time_in_seconds,
        frontend()->model->timestep_grouping,
        frontend()->total_number_of_synapses);
      CudaCheckError();
      //CudaSafeCall(cudaMemset(num_active_synapses, 0, sizeof(int)));
      //CudaSafeCall(cudaMemset(num_activated_neurons, 0, sizeof(int)));
      }
      
    }
      
    __device__ void get_active_synapses(
      spiking_synapses_data_struct* synaptic_data,
      spiking_neurons_data_struct* neuron_data,
      int timestep_group_index,
      int preneuron_idx,
      bool is_input)
    {
      int synapse_count = neuron_data->per_neuron_efferent_synapse_count[preneuron_idx];
      atomicMax(synaptic_data->num_active_synapses, synapse_count);
      int pos = atomicAdd(synaptic_data->num_activated_neurons, 1);
      synaptic_data->active_synapse_counts[pos] = synapse_count;
      synaptic_data->active_presynaptic_neuron_indices[pos] = CORRECTED_PRESYNAPTIC_ID(preneuron_idx, is_input);
      synaptic_data->group_indices[pos] = timestep_group_index;
    };
      

    __global__ void activate_synapses(
        spiking_neurons_data_struct* input_neuron_data,
        spiking_neurons_data_struct* neuron_data,
        int bufferloc,
        int buffersize,
        neuron_inputs_struct neuron_inputs,
        int* postsynaptic_neuron_indices,
        int* presynaptic_neuron_indices,
        float* synaptic_efficacies_or_weights,
        float* weight_scaling_constants,
        int* d_delays,
        int num_syn_labels,
        int * d_syn_labels,
        float timestep,
        float current_time_in_seconds,
        int timestep_grouping,
        int total_number_of_synapses)
    {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      int bufsize = input_neuron_data->neuron_spike_time_bitbuffer_bytesize[0];
      if (idx == 0){
        neuron_inputs.bufferloc[0] = (bufferloc + timestep_grouping) % buffersize;
      }
      while (idx < total_number_of_synapses){
        int preid = presynaptic_neuron_indices[idx];
        int postid = postsynaptic_neuron_indices[idx];
        bool is_input = PRESYNAPTIC_IS_INPUT(preid);
        int corr_preid = CORRECTED_PRESYNAPTIC_ID(preid, is_input);
        uint8_t* pre_bitbuffer = is_input ? input_neuron_data->neuron_spike_time_bitbuffer : neuron_data->neuron_spike_time_bitbuffer;
        // Looping over timesteps
        for (int g=0; g < timestep_grouping; g++){

          // Bit Indexing to detect spikes
          int postbitloc = ((int)roundf(current_time_in_seconds / timestep) + g) % (bufsize*8);
          int prebitloc = postbitloc - d_delays[idx] + timestep_grouping;
          prebitloc = (prebitloc < 0) ? (bufsize*8 + prebitloc) : prebitloc;


          // On Pre Update Synapse
          if (pre_bitbuffer[corr_preid*bufsize + (prebitloc / 8)] & (1 << (prebitloc % 8))){
            int targetloc = (bufferloc + timestep_grouping + g) % buffersize;
            int syn_label = d_syn_labels[idx];
            float weightinput = weight_scaling_constants[idx]*synaptic_efficacies_or_weights[idx];
            atomicAdd(&neuron_inputs.circular_input_buffer[targetloc*neuron_inputs.input_buffersize + syn_label + postid*num_syn_labels], weightinput);
          }
        }


        idx += blockDim.x * gridDim.x;
      }

      __syncthreads();
      if ((threadIdx.x + blockIdx.x * blockDim.x) == 0){
        num_activated_neurons[0] = 0;
      }
    }

      __device__ float spiking_current_injection_kernel(
  spiking_synapses_data_struct* synaptic_data,
  spiking_neurons_data_struct* neuron_data,
  float current_membrane_voltage,
  float current_time_in_seconds,
  float timestep,
  float multiplication_to_volts,
  int timestep_grouping,
  int idx,
  int g){
        return 0.0f;
      };

  }
}
