#include "VoltageActivityMonitor.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, VoltageActivityMonitor);

namespace Backend {
  namespace Dummy {
    void VoltageActivityMonitor::prepare() {
      ActivityMonitor::prepare();
    }

    void VoltageActivityMonitor::reset_state() {
      ActivityMonitor::reset_state();
    }
    
    void VoltageActivityMonitor::collect_measurement(unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping){
    }

    void VoltageActivityMonitor::copy_data_to_host() {
    }
  }
}
