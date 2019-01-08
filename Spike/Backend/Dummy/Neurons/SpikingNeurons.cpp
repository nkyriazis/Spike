#include "SpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, SpikingNeurons);

namespace Backend {
  namespace Dummy {
    SpikingNeurons::SpikingNeurons() {
    }

    void SpikingNeurons::prepare() {
      Neurons::prepare();
    }

    void SpikingNeurons::reset_state() {
      Neurons::reset_state();
    }

    void SpikingNeurons::state_update
    (unsigned int current_time_in_timesteps, float timestep) {
    }

  } // namespace Dummy
} // namespace Backend

