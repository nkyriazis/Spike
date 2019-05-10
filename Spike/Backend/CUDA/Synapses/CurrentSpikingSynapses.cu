// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/CurrentSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, CurrentSpikingSynapses);

namespace Backend {
  namespace CUDA {
    __device__ injection_kernel current_device_kernel = current_spiking_current_injection_kernel;
    
    CurrentSpikingSynapses::~CurrentSpikingSynapses(){
      CudaSafeCall(cudaFree(d_decay_factors));
    }
    void CurrentSpikingSynapses::prepare() {
      SpikingSynapses::prepare();

      // Carry out remaining device actions
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();

      current_spiking_synapses_data_struct temp_synaptic_data;
      memcpy(&temp_synaptic_data, synaptic_data, sizeof(spiking_synapses_data_struct));
      free(synaptic_data);
      synaptic_data = new current_spiking_synapses_data_struct();
      memcpy(synaptic_data, &temp_synaptic_data, sizeof(spiking_synapses_data_struct));
      current_spiking_synapses_data_struct* this_synaptic_data = static_cast<current_spiking_synapses_data_struct*>(synaptic_data);
      this_synaptic_data->synapse_type = CURRENT;
      CudaSafeCall(cudaMemcpy(
        d_synaptic_data,
        synaptic_data,
        sizeof(current_spiking_synapses_data_struct), cudaMemcpyHostToDevice));
    }
    
    void CurrentSpikingSynapses::allocate_device_pointers() {

      CudaSafeCall(cudaMalloc((void **)&d_decay_factors, sizeof(float)*frontend()->number_of_parameter_labels));
      CudaSafeCall(cudaFree(d_synaptic_data));
      CudaSafeCall(cudaMalloc((void **)&d_synaptic_data, sizeof(current_spiking_synapses_data_struct)));
      CudaSafeCall(cudaMemcpyFromSymbol(
            &host_injection_kernel,
            current_device_kernel,
            sizeof(injection_kernel)));
    }
    
    void CurrentSpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      std::vector<float> decay_factors;
      for (int p=0; p < frontend()->number_of_parameter_labels; p++)
        decay_factors.push_back(expf(-frontend()->model->timestep / frontend()->decay_terms_tau[p]));

      CudaSafeCall(cudaMemcpy(
        d_decay_factors,
        decay_factors.data(),
        sizeof(float)*frontend()->number_of_parameter_labels, cudaMemcpyHostToDevice));
    }

    void CurrentSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
    }

    void CurrentSpikingSynapses::state_update(unsigned int current_time_in_timesteps, float timestep) {
      SpikingSynapses::state_update(current_time_in_timesteps, timestep);
    }
    
    /* KERNELS BELOW */
    __device__ float current_spiking_current_injection_kernel(
        spiking_synapses_data_struct* in_synaptic_data,
        spiking_neurons_data_struct* neuron_data,
        float multiplication_to_volts,
        float current_membrane_voltage,
        unsigned int current_time_in_timesteps,
        float timestep,
        int idx,
        int g){
      /*
      current_spiking_synapses_data_struct* synaptic_data = (current_spiking_synapses_data_struct*) in_synaptic_data;
       
      int total_number_of_neurons =  neuron_data->total_number_of_neurons;
      int bufferloc = ((current_time_in_timesteps + g) % synaptic_data->neuron_inputs.temporal_buffersize)*synaptic_data->neuron_inputs.input_buffersize;
      float total_current = 0.0f;
        for (int syn_label = 0; syn_label < synaptic_data->num_syn_labels; syn_label++){
          float decay_term_value = synaptic_data->decay_terms_tau[syn_label];
          float decay_factor = expf(- timestep / decay_term_value);
          float synaptic_current = synaptic_data->neuron_wise_current_trace[total_number_of_neurons*syn_label + idx];
          // Update the synaptic conductance
          synaptic_current *= decay_factor;
          synaptic_current += synaptic_data->neuron_inputs.circular_input_buffer[bufferloc + syn_label + idx*synaptic_data->num_syn_labels];
          // Reset the conductance update
          synaptic_data->neuron_inputs.circular_input_buffer[bufferloc + syn_label*total_number_of_neurons + idx] = 0.0f;
          total_current += synaptic_current;
          synaptic_data->neuron_wise_current_trace[total_number_of_neurons*syn_label + idx] = synaptic_current;
    
        }
        
        return total_current*multiplication_to_volts;
        */
    };
  }
}
  
