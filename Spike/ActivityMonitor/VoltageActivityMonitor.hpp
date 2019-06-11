#ifndef VoltageActivityMonitor_H
#define VoltageActivityMonitor_H


#include "../ActivityMonitor/ActivityMonitor.hpp"

class VoltageActivityMonitor; // forward definition

namespace Backend {
  class VoltageActivityMonitor : public virtual ActivityMonitor {
  public:
    SPIKE_ADD_BACKEND_FACTORY(VoltageActivityMonitor);
    virtual void collect_measurement(unsigned int current_time_in_timesteps, float timestep) = 0;
    virtual void copy_data_to_host() = 0;
  };
}

class VoltageActivityMonitor : public ActivityMonitor {
public:
  SPIKE_ADD_BACKEND_GETSET(VoltageActivityMonitor,
                           ActivityMonitor);
  void init_backend(Context* ctx = _global_ctx) override;
  
  // Constructor/Destructor
  VoltageActivityMonitor(SpikingNeurons * neurons_parameter, int neuron_id);
  ~VoltageActivityMonitor() override = default;
      
  SpikingNeurons* neurons = nullptr;
  int neuron_id = -1;
  int num_measurements = 0;
  float * neuron_measurements = nullptr;

  void state_update(unsigned int current_time_in_timesteps, float timestep) override;
  void final_update(unsigned int current_time_in_timesteps, float timestep) override;
  void reset_state() override;
  void save_measurements_as_txt(string path, string prefix);
  void save_measurements_as_binary(string path, string prefix);

private:
  std::shared_ptr<::Backend::VoltageActivityMonitor> _backend;
};

#endif
