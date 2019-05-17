#ifndef LIFSpikingNeurons_H
#define LIFSpikingNeurons_H

#include "SpikingNeurons.hpp"


struct lif_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
  lif_spiking_neuron_parameters_struct() : somatic_capacitance_Cm(0.0f), somatic_leakage_conductance_g0(0.0f), background_current(0.0f), resting_potential_v0(-0.074f), after_spike_reset_potential_vreset(-0.074f), threshold_for_action_potential_spike(0.03f), absolute_refractory_period(0.002f) { spiking_neuron_parameters_struct(); }

  bool set_init_membrane = false;
  float membrane_potential_range[2];
  float resting_potential_v0;
  float after_spike_reset_potential_vreset;
  float threshold_for_action_potential_spike;
  float absolute_refractory_period;
  float somatic_capacitance_Cm;
  float somatic_leakage_conductance_g0;
  float background_current;

  bool adaptation = false;
  float adaptation_tau = 0.100;
  float adaptation_reversal_potential = -0.080;
  float adaptation_strength = 0.1;

};

class LIFSpikingNeurons; // forward definition

namespace Backend {
  class LIFSpikingNeurons : public virtual SpikingNeurons {
  public:
    SPIKE_ADD_BACKEND_FACTORY(LIFSpikingNeurons);
  };
}

class LIFSpikingNeurons : public SpikingNeurons {
public:
  // Constructor/Destructor
  LIFSpikingNeurons();
  ~LIFSpikingNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(LIFSpikingNeurons, SpikingNeurons);
  
  // Details of sub-populations
  vector<int> neuron_labels;
  vector<float> membrane_potentials_v;
  
  vector<float> after_spike_reset_potentials_vreset;
  vector<float> resting_potentials_v0;
  vector<float> spiking_thresholds_vthresh;
  vector<float> membrane_time_constants_tau_m;
  vector<float> membrane_resistances_R;
  vector<float> background_currents;
  vector<float> refractory_periods;
  vector<bool> adaptations;
  vector<float> adaptation_taus;
  vector<float> adaptation_reversal_potentials;
  vector<float> adaptation_strengths;
  
  int AddGroup(neuron_parameters_struct * group_params) override;
  void prepare_backend_early() override;

private:
  std::shared_ptr<::Backend::LIFSpikingNeurons> _backend;
};

#endif
