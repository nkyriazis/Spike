#include "CurrentActivityMonitor.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, CurrentActivityMonitor);

namespace Backend {
  namespace Dummy {
    void CurrentActivityMonitor::prepare() {
      ActivityMonitor::prepare();
    }

    void CurrentActivityMonitor::reset_state() {
      ActivityMonitor::reset_state();
    }
    
    void CurrentActivityMonitor::collect_measurement(unsigned int current_time_in_timesteps, float timestep){
    }

    void CurrentActivityMonitor::copy_data_to_host() {
    }
  }
}
