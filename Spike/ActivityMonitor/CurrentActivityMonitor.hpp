#ifndef CurrentActivityMonitor_H
#define CurrentActivityMonitor_H


#include "Spike/Synapses/CurrentSpikingSynapses.hpp"
#include "../ActivityMonitor/ActivityMonitor.hpp"

class CurrentActivityMonitor; // forward definition

namespace Backend {
  class CurrentActivityMonitor : public virtual ActivityMonitor {
  public:
    SPIKE_ADD_BACKEND_FACTORY(CurrentActivityMonitor);
    virtual void collect_measurement(unsigned int current_time_in_timesteps, float timestep) = 0;
    virtual void copy_data_to_host() = 0;
  };
}

class CurrentActivityMonitor : public ActivityMonitor {
public:
  SPIKE_ADD_BACKEND_GETSET(CurrentActivityMonitor,
                           ActivityMonitor);
  void init_backend(Context* ctx = _global_ctx) override;
  
  // Constructor/Destructor
  CurrentActivityMonitor(CurrentSpikingSynapses * synapses, int neuron_id, int label_id);
  ~CurrentActivityMonitor() override = default;
      
  CurrentSpikingSynapses* synapses = nullptr;
  int neuron_id = -1;
  int label_id = -1;
  int num_measurements = 0;
  float * measurements = nullptr;

  void state_update(unsigned int current_time_in_timesteps, float timestep) override;
  void final_update(unsigned int current_time_in_timesteps, float timestep) override;
  void reset_state() override;
  void save_measurements_as_txt(string path, string prefix);
  void save_measurements_as_binary(string path, string prefix);

private:
  std::shared_ptr<::Backend::CurrentActivityMonitor> _backend;
};

#endif
