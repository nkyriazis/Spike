#include "SpikingModel.hpp"

#include "../Neurons/InputSpikingNeurons.hpp"
#include "Spike/Helpers/TerminalHelpers.hpp"
#include "Spike/Backend/Context.hpp"


// SpikingModel Constructor
SpikingModel::SpikingModel () {
}


// SpikingModel Destructor
SpikingModel::~SpikingModel () {
}


void SpikingModel::SetTimestep(float timestep_parameter){
  if ((spiking_synapses == nullptr) || (spiking_synapses->total_number_of_synapses == 0)) {
    timestep = timestep_parameter;
  } else {
    print_message_and_exit("You must set the timestep before creating any synapses.");
  }
}

  /*
int SpikingModel::AddSynapseGroup(int presynaptic_group_id, 
              int postsynaptic_group_id, 
              synapse_parameters_struct * synapse_params) {
  if (spiking_synapses == nullptr) print_message_and_exit("Please set synapse pointer before adding synapses.");

  int groupID = spiking_synapses->AddGroup(presynaptic_group_id, 
              postsynaptic_group_id, 
              spiking_neurons,
              input_spiking_neurons,
              timestep,
              synapse_params);

  return(groupID);
}


void SpikingModel::AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, 
              synapse_parameters_struct * synapse_params) {

  for (int i = 0; i < input_spiking_neurons->total_number_of_groups; i++) {

    AddSynapseGroup(CORRECTED_PRESYNAPTIC_ID(i, true), 
              postsynaptic_group_id,
              synapse_params);

  }

}
              */

void SpikingModel::AddNeuronType(SpikingNeurons * neuron_model) {
  // Adds a neuron type to the list of neurons being simulated
  spiking_neuron_vec.push_back(neuron_model);
}

void SpikingModel::AddPlasticityRule(STDPPlasticity * plasticity_rule){
  // Adds the new STDP rule to the vector of STDP Rule
  plasticity_rule_vec.push_back(plasticity_rule);
}

void SpikingModel::AddActivityMonitor(ActivityMonitor * activityMonitor){
  // Adds the activity monitor to the vector
  monitors_vec.push_back(activityMonitor);
}

void SpikingModel::finalise_model() {
  if (!model_complete){
    printf("\n-----------\n");
    printf("---SPIKE---\n");
    printf("-----------\n\n");
    model_complete = true;
    
    // If any component does not exist, create at least a stand-in
    if (!spiking_synapses)
      spiking_synapses = new SpikingSynapses();
    

    timestep_grouping = spiking_synapses->minimum_axonal_delay_in_timesteps;
    // Don't let the total timestep grouping to exceed 2ms
    /*
    if (timestep_grouping * timestep > 0.001)
      timestep_grouping = (int)round(0.001f / timestep);
      */
    
    // Outputting Network Overview
    printf("Building Model with:\n");
    if (spiking_neuron_vec.size() > 0)
      printf("  %d Neuron Types(s)\n", (int)spiking_neuron_vec.size());
    for (int n = 0; n < spiking_neuron_vec.size(); n++)
      printf("    %d: %d Neuron(s)\n", (int)spiking_neuron_vec.size(), spiking_neuron_vec[n]->total_number_of_neurons);

    if (spiking_synapses->total_number_of_synapses > 0)
      printf("  %d Synapse(s)\n", spiking_synapses->total_number_of_synapses);
    if (plasticity_rule_vec.size() > 0)
      printf("  %d Plasticity Rule(s)\n", (int)plasticity_rule_vec.size());
    if (monitors_vec.size() > 0)
      printf("  %d Activity Monitor(s)\n", (int)monitors_vec.size());
    printf("\n");


    spiking_synapses->model = this;
    for (int n = 0; n < spiking_neuron_vec.size(); n++){
      spiking_neuron_vec[n]->model = this;
    }
    for (int plasticity_id = 0; plasticity_id < plasticity_rule_vec.size(); plasticity_id++){
      plasticity_rule_vec[plasticity_id]->model = this;
    }
    for (int monitor_id = 0; monitor_id < monitors_vec.size(); monitor_id++){
      monitors_vec[monitor_id]->model = this;
    }
    init_backend();
    reset_state();
  }
}
  

void SpikingModel::init_backend() {

  Backend::init_global_context();
  context = Backend::get_current_context();

  #ifndef SILENCE_MODEL_SETUP
  TimerWithMessages* timer = new TimerWithMessages("Setting Up Network...\n");
  #endif

  context->params.threads_per_block_neurons = 32;
  context->params.threads_per_block_synapses = 32;

  // NB All these also call prepare_backend for the initial state:
  for (int n = 0; n < spiking_neuron_vec.size(); n++){
    spiking_neuron_vec[n]->init_backend(context);
  }
  spiking_synapses->init_backend(context);
  for (int plasticity_id = 0; plasticity_id < plasticity_rule_vec.size(); plasticity_id++){
    plasticity_rule_vec[plasticity_id]->init_backend(context);
  }
  for (int monitor_id = 0; monitor_id < monitors_vec.size(); monitor_id++){
    monitors_vec[monitor_id]->init_backend(context);
  }

  #ifndef SILENCE_MODEL_SETUP
  timer->stop_timer_and_log_time_and_message("Network set up.", true);
  #endif
}


void SpikingModel::prepare_backend() {
  spiking_synapses->prepare_backend();
  context->params.maximum_axonal_delay_in_timesteps = spiking_synapses->maximum_axonal_delay_in_timesteps;
  
  for (int n = 0; n < spiking_neuron_vec.size(); n++){
    spiking_neuron_vec[n]->init_backend(context);
  }
  for (int plasticity_id = 0; plasticity_id < plasticity_rule_vec.size(); plasticity_id++){
    plasticity_rule_vec[plasticity_id]->prepare_backend();
  }
  for (int monitor_id = 0; monitor_id < monitors_vec.size(); monitor_id++){
    monitors_vec[monitor_id]->prepare_backend();
  }
}


void SpikingModel::reset_state() {
  finalise_model();

  spiking_synapses->reset_state();
  for (int n = 0; n < spiking_neuron_vec.size(); n++){
    spiking_neuron_vec[n]->reset_state();
  }
  for (int plasticity_id = 0; plasticity_id < plasticity_rule_vec.size(); plasticity_id++){
    plasticity_rule_vec[plasticity_id]->reset_state();
  }
}

void SpikingModel::reset_time() {
  current_time_in_seconds = 0.0f;
  current_time_in_timesteps = 0;
}


void SpikingModel::perform_per_step_model_instructions(bool plasticity_on){
  
  for (int n = 0; n < spiking_neuron_vec.size(); n++){
    spiking_neuron_vec[n]->state_update(current_time_in_timesteps, timestep);
  }
  
  if (plasticity_on){
    for (int plasticity_id = 0; plasticity_id < plasticity_rule_vec.size(); plasticity_id++)
      plasticity_rule_vec[plasticity_id]->state_update(current_time_in_timesteps, timestep);
  }

  spiking_synapses->state_update(current_time_in_timesteps, timestep);
  
  for (int monitor_id = 0; monitor_id < monitors_vec.size(); monitor_id++)
    monitors_vec[monitor_id]->state_update(current_time_in_timesteps, timestep);

}

void SpikingModel::run(float seconds, bool plasticity_on){
  // Finalise the model if not already done
  finalise_model();
  // Calculate the number of computational steps we need to do
  int number_of_timesteps = ceil(seconds / timestep);
  int number_of_steps = ceil(number_of_timesteps / timestep_grouping);

  printf("Running model for %f units of time (%d Timesteps) \n", seconds, (number_of_steps*timestep_grouping));

  // Run the simulation for the given number of steps
  for (int s = 0; s < number_of_steps; s++){
    current_time_in_seconds = current_time_in_timesteps*timestep;
    perform_per_step_model_instructions(plasticity_on);
    current_time_in_timesteps += timestep_grouping;
  }

  // Carry out any final checks and outputs from recording electrodes
  for (int monitor_id = 0; monitor_id < monitors_vec.size(); monitor_id++)
    monitors_vec[monitor_id]->final_update(current_time_in_timesteps, timestep);

}

