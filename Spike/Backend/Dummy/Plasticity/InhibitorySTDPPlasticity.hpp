#pragma once

#include "Spike/Plasticity/InhibitorySTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"

namespace Backend {
  namespace Dummy {
    class InhibitorySTDPPlasticity : public virtual ::Backend::Dummy::STDPPlasticity,
                           public virtual ::Backend::InhibitorySTDPPlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(InhibitorySTDPPlasticity);

      void prepare() override;
      void reset_state() override;

      void apply_stdp_to_synapse_weights(unsigned int current_time_in_timesteps, float timestep) override;
    };
  }
}
