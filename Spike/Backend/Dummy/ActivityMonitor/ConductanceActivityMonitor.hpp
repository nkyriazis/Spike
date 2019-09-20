#pragma once

#include "Spike/ActivityMonitor/ConductanceActivityMonitor.hpp"
#include "ActivityMonitor.hpp"

namespace Backend {
  namespace Dummy {
    class ConductanceActivityMonitor :
      public virtual ::Backend::Dummy::ActivityMonitor,
      public virtual ::Backend::ConductanceActivityMonitor {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(ConductanceActivityMonitor);

      void prepare() override;
      void reset_state() override;

      void collect_measurement
      (unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping) override;
      void copy_data_to_host() override;
    };
  }
}
