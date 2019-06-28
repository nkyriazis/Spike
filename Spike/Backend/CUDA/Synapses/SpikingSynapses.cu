// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikingSynapses);

namespace Backend {
  namespace CUDA {

    __device__ injection_kernel spiking_device_kernel = spiking_current_injection_kernel;

    SpikingSynapses::SpikingSynapses() {
    }

    SpikingSynapses::~SpikingSynapses() {
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
      synaptic_data->presynaptic_pointer_indices = presynaptic_pointer_indices;
      synaptic_data->postsynaptic_neuron_indices = postsynaptic_neuron_indices;
      synaptic_data->delays = delays;
      synaptic_data->synaptic_efficacies_or_weights = synaptic_efficacies_or_weights;


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
    (unsigned int current_time_in_timesteps, float timestep) {

      if (frontend()->total_number_of_synapses > 0){
      
      // Calculate buffer location
      int bufferloc = current_time_in_timesteps % buffersize;


      ::Backend::CUDA::SpikingNeurons* neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(frontend()->model->spiking_neurons->backend());
      assert(neurons_backend);
      ::Backend::CUDA::SpikingNeurons* input_neurons_backend =
        dynamic_cast<::Backend::CUDA::SpikingNeurons*>(frontend()->model->input_spiking_neurons->backend());
      assert(input_neurons_backend);

      activate_synapses<<<neurons_backend->number_of_neuron_blocks_per_grid, threads_per_block>>>(
          d_synaptic_data,
          neurons_backend->d_neuron_data,
          input_neurons_backend->d_neuron_data,
          bufferloc,
          timestep,
          current_time_in_timesteps,
          frontend()->model->timestep_grouping);
      CudaCheckError();
      }
      
    }
      
    __global__ void activate_synapses(
        spiking_synapses_data_struct* synaptic_data,
        spiking_neurons_data_struct* neurons_data,
        spiking_neurons_data_struct* in_neurons_data,
        int bufferloc,
        float timestep,
        unsigned int current_time_in_timesteps,
        int timestep_grouping)
    {
      int indx = threadIdx.x + blockIdx.x * blockDim.x;
      if (indx == 0){
        synaptic_data->num_activated_neurons[((current_time_in_timesteps / timestep_grouping) + 1) % 2] = 0;
      }
      while (indx < (synaptic_data->num_active_synapses[0]*synaptic_data->num_activated_neurons[((current_time_in_timesteps / timestep_grouping) % 2)])) {
 
        int pos = indx / synaptic_data->num_active_synapses[0]; 
        int idx = indx % synaptic_data->num_active_synapses[0]; 
        int synapse_count = synaptic_data->active_synapse_counts[pos];

        if (idx >= synapse_count){
          indx += blockDim.x * gridDim.x;
          continue;
        }

        int synapse_id = synaptic_data->active_synapse_starts[pos] + idx;
        int postneuron = synaptic_data->postsynaptic_neuron_indices[synapse_id];
        
        int targetloc = (bufferloc + synaptic_data->delays[synapse_id] + synaptic_data->group_indices[pos]) % synaptic_data->neuron_inputs.temporal_buffersize;
        int syn_label = synaptic_data->syn_labels[synapse_id];
        float weightinput = synaptic_data->synaptic_efficacies_or_weights[synapse_id];
        atomicAdd(&synaptic_data->neuron_inputs.circular_input_buffer[targetloc*synaptic_data->neuron_inputs.input_buffersize + syn_label + postneuron*synaptic_data->num_syn_labels], weightinput);
        indx += blockDim.x * gridDim.x;
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
