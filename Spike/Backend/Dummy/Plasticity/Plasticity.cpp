#include "Plasticity.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(Dummy, Plasticity);

namespace Backend {
  namespace Dummy {
    void Plasticity::prepare() {
    }

    void Plasticity::reset_state() {
    }

    void Plasticity::state_update(int current_time_in_timesteps, float timestep){}
  }
}
