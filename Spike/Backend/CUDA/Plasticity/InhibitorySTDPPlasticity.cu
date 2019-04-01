// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/InhibitorySTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, InhibitorySTDPPlasticity);

namespace Backend {
  namespace CUDA {
    InhibitorySTDPPlasticity::~InhibitorySTDPPlasticity() {
      CudaSafeCall(cudaFree(vogels_pre_memory_trace));
      CudaSafeCall(cudaFree(vogels_post_memory_trace));
      if (vogels_memory_trace_reset){
        free(vogels_memory_trace_reset);
      }
    }

    void InhibitorySTDPPlasticity::reset_state() {
      STDPPlasticity::reset_state();

      CudaSafeCall(cudaMemcpy((void*)vogels_pre_memory_trace,
                              (void*)vogels_memory_trace_reset,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void*)vogels_post_memory_trace,
                              (void*)vogels_memory_trace_reset,
                              sizeof(float)*total_number_of_plastic_synapses,
                              cudaMemcpyHostToDevice));
    }

    void InhibitorySTDPPlasticity::prepare() {
      STDPPlasticity::prepare();

      vogels_memory_trace_reset = (float*)malloc(sizeof(float)*total_number_of_plastic_synapses);
      for (int i=0; i < total_number_of_plastic_synapses; i++){
        vogels_memory_trace_reset[i] = 0.0f;
      }

      allocate_device_pointers();
    }

    void InhibitorySTDPPlasticity::allocate_device_pointers() {
      // The following doesn't do anything in original code...
      // ::Backend::CUDA::STDPPlasticity::allocate_device_pointers();

      CudaSafeCall(cudaMalloc((void **)&vogels_pre_memory_trace, sizeof(int)*total_number_of_plastic_synapses));
      CudaSafeCall(cudaMalloc((void **)&vogels_post_memory_trace, sizeof(int)*total_number_of_plastic_synapses));
    }

    void InhibitorySTDPPlasticity::apply_stdp_to_synapse_weights(unsigned int current_time_in_timesteps, float timestep) {

    // Vogels update rule requires a neuron wise memory trace. This must be updated upon neuron firing.
    vogels_apply_stdp_to_synapse_weights_kernel<<<neurons_backend->number_of_neuron_blocks_per_grid, neurons_backend->threads_per_block>>>
       (synapses_backend->postsynaptic_neuron_indices,
       synapses_backend->presynaptic_neuron_indices,
       synapses_backend->delays,
       neurons_backend->d_neuron_data,
       input_neurons_backend->d_neuron_data,
       synapses_backend->synaptic_efficacies_or_weights,
       vogels_pre_memory_trace,
       vogels_post_memory_trace,
       //frontend()->stdp_params->tau_istdp,
       expf(- timestep / frontend()->stdp_params->tau_istdp),
       *(frontend()->stdp_params),
       timestep,
       frontend()->model->timestep_grouping,
       current_time_in_timesteps,
       plastic_synapse_indices,
       total_number_of_plastic_synapses);
    CudaCheckError();
    }

    __global__ void vogels_apply_stdp_to_synapse_weights_kernel
          (int* d_postsyns,
           int* d_presyns,
           int* d_syndelays,
           spiking_neurons_data_struct* neuron_data,
           spiking_neurons_data_struct* input_neuron_data,
           float* d_synaptic_efficacies_or_weights,
           float* vogels_pre_memory_trace,
           float* vogels_post_memory_trace,
           float trace_decay,
           struct inhibitory_stdp_plasticity_parameters_struct stdp_vars,
           float timestep,
           int timestep_grouping,
           unsigned int current_time_in_timesteps,
           int* d_plastic_synapse_indices,
           size_t total_number_of_plastic_synapses){
      // Global Index
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      // Running though all neurons
      while (indx < total_number_of_plastic_synapses) {
        int idx = d_plastic_synapse_indices[indx];

        // Getting synapse details
        float vogels_pre_memory_trace_val = vogels_pre_memory_trace[indx];
        float vogels_post_memory_trace_val = vogels_post_memory_trace[indx];
        int postid = d_postsyns[idx];
        int preid = d_presyns[idx];
        int bufsize = input_neuron_data->neuron_spike_time_bitbuffer_bytesize[0];
        float old_synaptic_weight = d_synaptic_efficacies_or_weights[idx];
        float new_synaptic_weight = old_synaptic_weight;

        // Correcting for input vs output neuron types
        bool is_input = PRESYNAPTIC_IS_INPUT(preid);
        int corr_preid = CORRECTED_PRESYNAPTIC_ID(preid, is_input);
        uint8_t* pre_bitbuffer = is_input ? input_neuron_data->neuron_spike_time_bitbuffer : neuron_data->neuron_spike_time_bitbuffer;

        /*
        float* pre_last_spike_times = is_input ? input_neuron_data->last_spike_time_of_each_neuron : neuron_data->last_spike_time_of_each_neuron;
        //int pre_spike_g = -1;
        int pre_spike_g = ((int)roundf(pre_last_spike_times[corr_preid] / timestep) - current_time_in_timesteps);
        int post_spike_g = ((int)roundf(neuron_data->last_spike_time_of_each_neuron[postid] / timestep) - current_time_in_timesteps);
        if (pre_spike_g >= timestep_grouping)
          pre_spike_g *= -1;

        vogels_post_memory_trace_val *= expf(-(timestep_grouping*timestep) / trace_decay);
        vogels_pre_memory_trace_val *= expf(-(timestep_grouping*timestep) / trace_decay);

        // Change this if nearest only
        vogels_post_memory_trace_val += (post_spike_g >= 0) ? expf(-((timestep_grouping - post_spike_g - 1)*timestep) / trace_decay) : 0.0f;
        vogels_pre_memory_trace_val += (pre_spike_g >= 0) ? expf(-((timestep_grouping - pre_spike_g - 1)*timestep) / trace_decay) : 0.0f;
          
        float syn_update_val = 0.0f; 
        //old_synaptic_weight = new_synaptic_weight;
        // OnPre Weight Update
        if (pre_spike_g >= 0){
          float temp_post_trace = vogels_post_memory_trace_val;
          temp_post_trace += (post_spike_g > pre_spike_g) ? -expf(-((timestep_grouping - post_spike_g - 1)*timestep) / trace_decay): 0.0f;
          temp_post_trace *= (1.0f / (expf(-(timestep_grouping - pre_spike_g - 1)*timestep / trace_decay))); 
          syn_update_val += stdp_vars.learningrate*(temp_post_trace);
          syn_update_val += - stdp_vars.learningrate*(2.0*stdp_vars.targetrate*stdp_vars.tau_istdp);
          //syn_update_val -= stdp_vars.lambda * stdp_vars.alpha * old_synaptic_weight * temp_post_trace;
        }
        // OnPost Weight Update
        if (post_spike_g >= 0){
          float temp_pre_trace = vogels_pre_memory_trace_val;
          temp_pre_trace += (pre_spike_g > post_spike_g) ? -expf(-((timestep_grouping - pre_spike_g - 1)*timestep) / trace_decay): 0.0f;
          temp_pre_trace *= (1.0f / (expf(-(timestep_grouping - post_spike_g - 1)*timestep / trace_decay))); 
          syn_update_val += stdp_vars.learningrate*(temp_pre_trace);
          //syn_update_val += stdp_vars.lambda * (stdp_vars.w_max - old_synaptic_weight) * temp_pre_trace;
        }

        new_synaptic_weight = old_synaptic_weight + syn_update_val;
        if (new_synaptic_weight < 0.0f)
          new_synaptic_weight = 0.0f;
        
        // Weight Update
        d_synaptic_efficacies_or_weights[idx] = new_synaptic_weight;

        // Correctly set the trace values
        vogels_pre_memory_trace[indx] = vogels_pre_memory_trace_val;
        vogels_post_memory_trace[indx] = vogels_post_memory_trace_val;

        indx += blockDim.x * gridDim.x;
        */
        // Looping over timesteps
        for (int g=0; g < timestep_grouping; g++){	
          // Decaying STDP traces
          vogels_pre_memory_trace_val *= trace_decay;
          vogels_post_memory_trace_val *= trace_decay;
          
          // Bit Indexing to detect spikes
          int postbitloc = (current_time_in_timesteps + g) % (bufsize*8);
          int prebitloc = postbitloc - d_syndelays[idx];
          prebitloc = (prebitloc < 0) ? (bufsize*8 + prebitloc) : prebitloc;

          // OnPre Trace Update
          if (pre_bitbuffer[corr_preid*bufsize + (prebitloc / 8)] & (1 << (prebitloc % 8))){
            vogels_pre_memory_trace_val += 1.0f;
          }
          // OnPost Trace Update
          if (neuron_data->neuron_spike_time_bitbuffer[postid*bufsize + (postbitloc / 8)] & (1 << (postbitloc % 8))){
            vogels_post_memory_trace_val += 1.0f;
          }

          float syn_update_val = 0.0f; 
          old_synaptic_weight = new_synaptic_weight;
          // OnPre Weight Update
          if (pre_bitbuffer[corr_preid*bufsize + (prebitloc / 8)] & (1 << (prebitloc % 8))){
            syn_update_val += stdp_vars.learningrate*(vogels_post_memory_trace_val);
            syn_update_val += -stdp_vars.learningrate*(2.0*stdp_vars.targetrate*stdp_vars.tau_istdp);
          }
          // OnPost Weight Update
          if (neuron_data->neuron_spike_time_bitbuffer[postid*bufsize + (postbitloc / 8)] & (1 << (postbitloc % 8))){
            syn_update_val += stdp_vars.learningrate*(vogels_pre_memory_trace_val);
          }

          new_synaptic_weight = old_synaptic_weight + syn_update_val;
          if (new_synaptic_weight < 0.0f)
            new_synaptic_weight = 0.0f;
        }

        if (new_synaptic_weight > stdp_vars.w_max)
          new_synaptic_weight = stdp_vars.w_max;

        // Weight Update
        d_synaptic_efficacies_or_weights[idx] = new_synaptic_weight;
        
        // Correctly set the trace values
        vogels_pre_memory_trace[indx] = vogels_pre_memory_trace_val;
        vogels_post_memory_trace[indx] = vogels_post_memory_trace_val;

        indx += blockDim.x * gridDim.x;
      }

    }
  }
}
