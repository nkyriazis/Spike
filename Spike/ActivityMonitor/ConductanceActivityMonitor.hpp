#ifndef ConductanceActivityMonitor_H
#define ConductanceActivityMonitor_H


#include "../ActivityMonitor/ActivityMonitor.hpp"

class ConductanceActivityMonitor; // forward definition

namespace Backend {
  class ConductanceActivityMonitor : public virtual ActivityMonitor {
  public:
    SPIKE_ADD_BACKEND_FACTORY(ConductanceActivityMonitor);
    virtual void collect_measurement(unsigned int current_time_in_timesteps, float timestep) = 0;
    virtual void copy_data_to_host() = 0;
  };
}

class ConductanceActivityMonitor : public ActivityMonitor {
public:
  SPIKE_ADD_BACKEND_GETSET(ConductanceActivityMonitor,
                           ActivityMonitor);
  void init_backend(Context* ctx = _global_ctx) override;
  
  // Constructor/Destructor
  ConductanceActivityMonitor(ConductanceSpikingSynapses * synapses, int neuron_id);
  ~ConductanceActivityMonitor() override = default;
      
  ConductanceSpikingSynapses* synapses = nullptr;
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
  std::shared_ptr<::Backend::ConductanceActivityMonitor> _backend;
};

#endif
