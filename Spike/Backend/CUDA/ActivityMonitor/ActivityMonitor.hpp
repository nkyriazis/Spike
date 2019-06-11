#pragma once

#include "Spike/ActivityMonitor/ActivityMonitor.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class ActivityMonitor : public virtual ::Backend::ActivityMonitor {
    public:
      using ::Backend::ActivityMonitor::frontend;

      void prepare() override;
      void reset_state() override;

    };
  }
}
