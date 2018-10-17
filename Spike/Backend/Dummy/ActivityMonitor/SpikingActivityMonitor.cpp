#include "SpikingActivityMonitor.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, SpikingActivityMonitor);

namespace Backend {
  namespace Dummy {
    void SpikingActivityMonitor::prepare() {
      ActivityMonitor::prepare();
    }

    void SpikingActivityMonitor::reset_state() {
      ActivityMonitor::reset_state();
    }
    
    void SpikingActivityMonitor::copy_spikecount_to_front() {
    }

    void SpikingActivityMonitor::copy_spikes_to_front() {
    }

    void SpikingActivityMonitor::collect_spikes_for_timestep
    (int current_time_in_timesteps, float timestep) {
    }
  }
}
