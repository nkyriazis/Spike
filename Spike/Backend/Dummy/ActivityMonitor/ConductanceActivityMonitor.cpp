#include "ConductanceActivityMonitor.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, ConductanceActivityMonitor);

namespace Backend {
  namespace Dummy {
    void ConductanceActivityMonitor::prepare() {
      ActivityMonitor::prepare();
    }

    void ConductanceActivityMonitor::reset_state() {
      ActivityMonitor::reset_state();
    }
    
    void ConductanceActivityMonitor::collect_measurement(unsigned int current_time_in_timesteps, float timestep){
    }

    void ConductanceActivityMonitor::copy_data_to_host() {
    }
  }
}
