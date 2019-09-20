#pragma once

#include "Spike/ActivityMonitor/CurrentActivityMonitor.hpp"
#include "ActivityMonitor.hpp"

namespace Backend {
  namespace Dummy {
    class CurrentActivityMonitor :
      public virtual ::Backend::Dummy::ActivityMonitor,
      public virtual ::Backend::CurrentActivityMonitor {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(CurrentActivityMonitor);

      void prepare() override;
      void reset_state() override;

      void collect_measurement
      (unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping) override;
      void copy_data_to_host() override;
    };
  }
}
