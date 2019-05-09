// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/Synapses.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, Synapses);

namespace Backend {
  namespace CUDA {
    Synapses::~Synapses() {
      CudaSafeCall(cudaFree(presynaptic_neuron_indices));
      CudaSafeCall(cudaFree(postsynaptic_neuron_indices));
      CudaSafeCall(cudaFree(temp_presynaptic_neuron_indices));
      CudaSafeCall(cudaFree(temp_postsynaptic_neuron_indices));
      CudaSafeCall(cudaFree(synaptic_efficacies_or_weights));
      CudaSafeCall(cudaFree(temp_synaptic_efficacies_or_weights));
      CudaSafeCall(cudaFree(weight_scaling_constants));
      CudaSafeCall(cudaFree(synapse_set_indices));
      CudaSafeCall(cudaFree(d_synaptic_data));
    }

    void Synapses::reset_state() {
    }

    void Synapses::allocate_device_pointers() {
      CudaSafeCall(cudaMalloc((void **)&presynaptic_neuron_indices,
                              sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&postsynaptic_neuron_indices,
                              sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&synaptic_efficacies_or_weights,
                              sizeof(float)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&weight_scaling_constants,
                              sizeof(float)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&synapse_set_indices,
                              sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&d_synaptic_data,
                              sizeof(synapses_data_struct)));
    }


    void Synapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(presynaptic_neuron_indices,
                              frontend()->presynaptic_neuron_indices,
                              sizeof(int)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(postsynaptic_neuron_indices,
                              frontend()->postsynaptic_neuron_indices,
                              sizeof(int)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(synaptic_efficacies_or_weights,
                              frontend()->synaptic_efficacies_or_weights,
                              sizeof(float)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(weight_scaling_constants,
                              frontend()->weight_scaling_constants,
                              sizeof(float)*frontend()->total_number_of_synapses, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(synapse_set_indices,
                              frontend()->synapse_set_indices.data(),
                              sizeof(int)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
    }


    void Synapses::set_threads_per_block_and_blocks_per_grid(int threads) {
      threads_per_block.x = threads;
      cudaDeviceProp deviceProp;
      int deviceID;
      cudaGetDevice(&deviceID);
      cudaGetDeviceProperties(&deviceProp, deviceID);
      max_num_blocks_per_grid = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor / threads);
      int theoretical_number = (frontend()->total_number_of_synapses + threads) / threads;
      if (theoretical_number < max_num_blocks_per_grid)
        number_of_synapse_blocks_per_grid = dim3(theoretical_number);
      else
        number_of_synapse_blocks_per_grid = dim3(max_num_blocks_per_grid);
    }

    void Synapses::prepare() {
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();
      set_threads_per_block_and_blocks_per_grid(context->params.threads_per_block_synapses);

      random_state_manager_backend
        = dynamic_cast<::Backend::CUDA::RandomStateManager*>
        (frontend()->random_state_manager->backend());

      synaptic_data = new synapses_data_struct();
      CudaSafeCall(cudaMemcpy(d_synaptic_data,
                              synaptic_data,
                              sizeof(synapses_data_struct),
                              cudaMemcpyHostToDevice));

    }

    void Synapses::copy_to_frontend(){
      // If weights are on the device, copy them back
      if (synaptic_efficacies_or_weights){
        CudaSafeCall(cudaMemcpy(frontend()->synaptic_efficacies_or_weights,
              synaptic_efficacies_or_weights,
              sizeof(float)*frontend()->total_number_of_synapses,
              cudaMemcpyDeviceToHost));
      }
    }
    void Synapses::copy_to_backend(){
      // If weights are on the device, copy over them
      if (synaptic_efficacies_or_weights){
        CudaSafeCall(cudaMemcpy(synaptic_efficacies_or_weights,
              frontend()->synaptic_efficacies_or_weights,
              sizeof(float)*frontend()->total_number_of_synapses,
              cudaMemcpyHostToDevice));
      }
    }

  }
}

