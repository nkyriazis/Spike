#ifndef ActivityMonitor_H
#define ActivityMonitor_H

using namespace std;

class ActivityMonitor; // forward definition
namespace Backend {
  class ActivityMonitor;
}
#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Synapses/SpikingSynapses.hpp"


namespace Backend {
  class ActivityMonitor : public virtual SpikeBackendBase {
  public:
    SPIKE_ADD_BACKEND_FACTORY(ActivityMonitor);
  };
}

class ActivityMonitor : public virtual SpikeBase {
public:
  ~ActivityMonitor() override = default;

  void init_backend(Context* ctx = _global_ctx) override;
  SPIKE_ADD_BACKEND_GETSET(ActivityMonitor, SpikeBase);

  virtual void state_update(unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping) = 0;
  virtual void final_update(unsigned int current_time_in_timesteps, float timestep, unsigned int timestep_grouping) = 0;
  virtual void reset_state() = 0;

private:
  std::shared_ptr<::Backend::ActivityMonitor> _backend;
};

#endif
