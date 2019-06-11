#pragma once

#include "Spike/ActivityMonitor/VoltageActivityMonitor.hpp"
#include "ActivityMonitor.hpp"

namespace Backend {
  namespace Dummy {
    class VoltageActivityMonitor :
      public virtual ::Backend::Dummy::ActivityMonitor,
      public virtual ::Backend::VoltageActivityMonitor {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(VoltageActivityMonitor);

      void prepare() override;
      void reset_state() override;

      void collect_measurement
      (unsigned int current_time_in_timesteps, float timestep) override;
      void copy_data_to_host() override;
    };
  }
}
