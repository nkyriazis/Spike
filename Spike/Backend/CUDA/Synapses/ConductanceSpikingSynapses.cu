// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, ConductanceSpikingSynapses);

namespace Backend {
  namespace CUDA {
    __device__ injection_kernel conductance_device_kernel = conductance_spiking_current_injection_kernel;

    // ConductanceSpikingSynapses Destructor
    ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {
      CudaSafeCall(cudaFree(neuron_wise_conductance_trace));
      CudaSafeCall(cudaFree(d_synaptic_data));
      CudaSafeCall(cudaFree(d_decay_factors_g));
      CudaSafeCall(cudaFree(d_reversal_potentials_Vhat));
    }

    void ConductanceSpikingSynapses::prepare() {
      SpikingSynapses::prepare();


      // Carry out remaining device actions
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();

      conductance_spiking_synapses_data_struct temp_synaptic_data;
      memcpy(&temp_synaptic_data, synaptic_data, sizeof(spiking_synapses_data_struct));
      free(synaptic_data);
      synaptic_data = new conductance_spiking_synapses_data_struct();
      memcpy(synaptic_data, &temp_synaptic_data, sizeof(spiking_synapses_data_struct));
      conductance_spiking_synapses_data_struct* this_synaptic_data = static_cast<conductance_spiking_synapses_data_struct*>(synaptic_data); 
      this_synaptic_data->decay_factors_g = d_decay_factors_g;
      this_synaptic_data->reversal_potentials_Vhat = d_reversal_potentials_Vhat;
      this_synaptic_data->neuron_wise_conductance_trace = neuron_wise_conductance_trace;
      this_synaptic_data->synapse_type = CONDUCTANCE;
      CudaSafeCall(cudaMemcpy(
        d_synaptic_data,
        synaptic_data,
        sizeof(conductance_spiking_synapses_data_struct), cudaMemcpyHostToDevice));

    }

    void ConductanceSpikingSynapses::reset_state() {
      SpikingSynapses::reset_state();
      
      CudaSafeCall(cudaMemset(neuron_wise_conductance_trace, 0.0f, sizeof(float)*frontend()->number_of_parameter_labels*frontend()->post_neuron_set[0]->total_number_of_neurons));
    }


    void ConductanceSpikingSynapses::allocate_device_pointers() {
      // Set up per neuron conductances
      CudaSafeCall(cudaMalloc((void **)&neuron_wise_conductance_trace, sizeof(float)*frontend()->number_of_parameter_labels*frontend()->post_neuron_set[0]->total_number_of_neurons));


      CudaSafeCall(cudaMalloc((void **)&d_decay_factors_g, sizeof(float)*synaptic_data->num_parameter_sets));
      CudaSafeCall(cudaMalloc((void **)&d_reversal_potentials_Vhat, sizeof(float)*synaptic_data->num_parameter_sets));
      CudaSafeCall(cudaFree(d_synaptic_data));
      CudaSafeCall(cudaMalloc((void **)&d_synaptic_data, sizeof(conductance_spiking_synapses_data_struct)));
      CudaSafeCall(cudaMemcpyFromSymbol(
            &host_injection_kernel,
            conductance_device_kernel,
            sizeof(injection_kernel)));
    }

    void ConductanceSpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
      vector<float> decay_vals_g;
      for (int syn_label_indx = 0; syn_label_indx < synaptic_data->num_parameter_sets; syn_label_indx++)
        decay_vals_g.push_back((expf(-frontend()->model->timestep / frontend()->decay_terms_tau_g[syn_label_indx])));
      CudaSafeCall(cudaMemcpy(
        d_decay_factors_g,
        decay_vals_g.data(),
        sizeof(float)*synaptic_data->num_parameter_sets, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(
        d_reversal_potentials_Vhat,
        &(frontend()->reversal_potentials_Vhat[0]),
        sizeof(float)*synaptic_data->num_parameter_sets, cudaMemcpyHostToDevice));

      CudaSafeCall(cudaMemset(neuron_wise_conductance_trace, 0.0f, sizeof(float)*frontend()->number_of_parameter_labels*frontend()->post_neuron_set[0]->total_number_of_neurons));
    }



    /* STATE UPDATE */
    void ConductanceSpikingSynapses::state_update
    (unsigned int current_time_in_timesteps, float timestep) {
      SpikingSynapses::state_update(current_time_in_timesteps, timestep);
    }


    /* KERNELS BELOW */
    __device__ float conductance_spiking_current_injection_kernel(
        spiking_synapses_data_struct* in_synaptic_data,
        spiking_neurons_data_struct* neuron_data,
        float multiplication_to_volts,
        float current_membrane_voltage,
        unsigned int current_time_in_timesteps,
        float timestep,
        int idx,
        int g){
      /*
      
      conductance_spiking_synapses_data_struct* synaptic_data = (conductance_spiking_synapses_data_struct*) in_synaptic_data;
        
      int bufferloc = ((current_time_in_timesteps + g) % synaptic_data->neuron_inputs.temporal_buffersize)*synaptic_data->neuron_inputs.input_buffersize;
        
      float total_current = 0.0f;
      for (int syn_label = 0; syn_label < synaptic_data->num_synapse_groups; syn_label++){
        float decay_factor = synaptic_data->decay_factors_g[syn_label];
        float reversal_value = synaptic_data->reversal_potentials_Vhat[syn_label];
        float synaptic_conductance_g = synaptic_data->neuron_wise_conductance_trace[syn_label + idx*synaptic_data->num_synapse_groups];
        // Update the synaptic conductance
        synaptic_conductance_g = decay_factor*synaptic_conductance_g + synaptic_data->neuron_inputs.circular_input_buffer[bufferloc + syn_label + idx*synaptic_data->num_synapse_groups];
        // Reset the conductance update
        synaptic_data->neuron_inputs.circular_input_buffer[bufferloc + syn_label + idx*synaptic_data->num_synapse_groups] = 0.0f;
        total_current += synaptic_conductance_g*(reversal_value - current_membrane_voltage);
    
        synaptic_data->neuron_wise_conductance_trace[syn_label + idx*synaptic_data->num_synapse_groups] = synaptic_conductance_g;
    
      }
      return total_current*multiplication_to_volts;
      */
    };
  }
}
