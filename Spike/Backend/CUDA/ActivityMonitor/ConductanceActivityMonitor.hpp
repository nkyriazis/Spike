#pragma once

#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.hpp"

#include "Spike/ActivityMonitor/ConductanceActivityMonitor.hpp"
#include "ActivityMonitor.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class ConductanceActivityMonitor :
      public virtual ::Backend::CUDA::ActivityMonitor,
      public virtual ::Backend::ConductanceActivityMonitor {
    public:
      ~ConductanceActivityMonitor() override;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(ConductanceActivityMonitor);
      using ::Backend::ConductanceActivityMonitor::frontend;

      int max_num_measurements = 1000;
      int num_measurements = 0;
      float* measurements = nullptr;
      
      void prepare() override;
      void reset_state() override;

      void allocate_pointers_for_data();

      void copy_data_to_host() override;
      void collect_measurement(unsigned int current_time_in_timesteps, float timestep) override;
    
    private:
      ::ConductanceSpikingSynapses* synapses_frontend = nullptr;
      ::Backend::CUDA::ConductanceSpikingSynapses* synapses_backend = nullptr;

    };
  }
}
