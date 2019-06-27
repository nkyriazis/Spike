#ifndef SpikingNeurons_H
#define SpikingNeurons_H

#include <cstdio>

class SpikingNeurons; // Forward Definition
struct spiking_neuron_parameters_struct;

#include "Spike/Models/SpikingModel.hpp"
#include "Neurons.hpp"


struct spiking_neuron_parameters_struct : neuron_parameters_struct {
};

class SpikingNeurons; // forward definition

namespace Backend {
  class SpikingNeurons : public virtual Neurons {
  public:
    SPIKE_ADD_BACKEND_FACTORY(SpikingNeurons);
    virtual void state_update(unsigned int current_time_in_timesteps, float timestep) = 0;
  };
}

class SpikingNeurons : public Neurons {
public:
  // Constructor/Destructor
  SpikingNeurons();
  ~SpikingNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(SpikingNeurons, Neurons);
  void prepare_backend_early() override;

  // Host Pointers
  SpikingModel* model = nullptr;

  // Functions
  int AddGroup(neuron_parameters_struct * group_params) override;

  virtual void state_update(unsigned int current_time_in_timesteps, float timestep);

private:
  std::shared_ptr<::Backend::SpikingNeurons> _backend;
};

#endif
