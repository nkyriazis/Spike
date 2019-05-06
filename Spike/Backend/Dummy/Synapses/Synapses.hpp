#pragma once

#include "Spike/Synapses/Synapses.hpp"

namespace Backend {
  namespace Dummy {
    class Synapses : public virtual ::Backend::Synapses {
    public:
      ~Synapses() override = default;

      void prepare() override;
      void reset_state() override;

      void copy_to_frontend() override;
      void copy_to_backend() override;
    };
  } // namespace Dummy
} // namespace Backend

