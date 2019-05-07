#include "SpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

SpikingSynapses::~SpikingSynapses() {
#ifdef CRAZY_DEBUG
  std::cout << "SpikingSynapses::~SpikingSynapses\n";
#endif
  free(delays);

}

void SpikingSynapses::prepare_backend_early() {
  Synapses::prepare_backend_early();
  Synapses::sort_synapses();
  SpikingSynapses::sort_synapses();
}


void SpikingSynapses::sort_synapses(){
  
  int* temp_delay_array = (int*)malloc(total_number_of_synapses * sizeof(int));
  // Re-ordering arrays
  for (int s=0; s < total_number_of_synapses; s++){
    temp_delay_array[s] = delays[synapse_sort_indices[s]];
  }

  free(delays);
  delays = temp_delay_array;
}

int SpikingSynapses::AddGroup(int presynaptic_group_id, 
            int postsynaptic_group_id, 
            Neurons * pre_neurons,
            Neurons * post_neurons,
            float timestep,
            synapse_parameters_struct * synapse_params) {
  
  
  int groupID = Synapses::AddGroup(presynaptic_group_id, 
              postsynaptic_group_id, 
              pre_neurons,
              post_neurons,
              timestep,
              synapse_params);

  // First incrementing the synapses
  SpikingSynapses::increment_number_of_synapses(temp_number_of_synapses_in_last_group);

  spiking_synapse_parameters_struct * spiking_synapse_group_params = (spiking_synapse_parameters_struct*)synapse_params;

  // Convert delay range from time to number of timesteps
  int delay_range_in_timesteps[2] = {int(round(spiking_synapse_group_params->delay_range[0]/timestep)), int(round(spiking_synapse_group_params->delay_range[1]/timestep))};

  // Check delay range bounds greater than timestep
  if ((delay_range_in_timesteps[0] < 1) || (delay_range_in_timesteps[1] < 1)) {
    printf("%d\n", delay_range_in_timesteps[0]);
    printf("%d\n", delay_range_in_timesteps[1]);
#ifdef CRAZY_DEBUG
                // spiking_synapse_group_params->delay_range[0] = timestep;
                // spiking_synapse_group_params->delay_range[1] = timestep;
    printf("################### Delay range must be at least one timestep\n");
#else

        
    print_message_and_exit("Delay range must be at least one timestep.");
#endif
  }
  
  for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++){
    // Setup Delays
    float delayval = delay_range_in_timesteps[0];
    if (delay_range_in_timesteps[0] != delay_range_in_timesteps[1])
      delayval = delay_range_in_timesteps[0] + (delay_range_in_timesteps[1] - delay_range_in_timesteps[0]) * ((float)rand() / (RAND_MAX));
    delays[i] = round(delayval);
    if (spiking_synapse_group_params->connectivity_type == CONNECTIVITY_TYPE_PAIRWISE){
      if (spiking_synapse_group_params->pairwise_connect_delay.size() == temp_number_of_synapses_in_last_group){
        delays[i] = (int)round(spiking_synapse_group_params->pairwise_connect_delay[i + temp_number_of_synapses_in_last_group - total_number_of_synapses] / timestep);
        if (delays[i] < 1){
          print_message_and_exit("PAIRWISE CONNECTION ERROR: All delays must be greater than one timestep.");
        }
      } else if (spiking_synapse_group_params->pairwise_connect_delay.size() != 0) {
        print_message_and_exit("PAIRWISE CONNECTION ERROR: Delay vector length not as expected. Should be the same length as pre/post vecs.");
      }
    }
    
    // Ensure max/min delays are set correctly
    if (delays[i] > maximum_axonal_delay_in_timesteps) maximum_axonal_delay_in_timesteps = delays[i];
    if (delays[i] < minimum_axonal_delay_in_timesteps) minimum_axonal_delay_in_timesteps = delays[i];
  }

  return groupID;

}

void SpikingSynapses::increment_number_of_synapses(int increment) {
  delays = (int*)realloc(delays, total_number_of_synapses * sizeof(int));
}


void SpikingSynapses::state_update(unsigned int current_time_in_timesteps, float timestep) {
  backend()->state_update(current_time_in_timesteps, timestep);
}

void SpikingSynapses::save_connectivity_as_txt(std::string path, std::string prefix, int synapsegroupid){
  if (startid < 0)
    print_message_and_exit("Synapse saving error: Provide a non-zero synapse group id!\n");
  int startid = 0;
  int endid = total_number_of_synapses;
  if (synapsegroupid >= 0)
    endid = last_index_of_synapse_per_group[synapsegroupid];
  if ((synapsegroupid > 0) && (synapsegroupid < last_index_of_synapse_per_group.size())){
    startid = last_index_of_synapse_per_group[synapsegroupid - 1];
  }
  Synapses::save_connectivity_as_txt(path, prefix, synapsegroupid);
  std::ofstream delayfile;

  // Open output files
  delayfile.open((path + "/" + prefix + "SynapticDelays.txt"), std::ios::out | std::ios::binary);

  // Ensure weight data has been copied to frontend
  if (_backend)
    backend()->copy_to_frontend();

  // Send data to file
  for (int i = startid; i < endid; i++){
    delayfile << delays[synapse_reversesort_indices[i]] << std::endl;
  }

  // Close files
  delayfile.close();

};
// Ensure copied from device, then send
void SpikingSynapses::save_connectivity_as_binary(std::string path, std::string prefix, int synapsegroupid){
  if (startid < 0)
    print_message_and_exit("Synapse saving error: Provide a non-zero synapse group id!\n");
  int startid = 0;
  int endid = total_number_of_synapses;
  if (synapsegroupid >= 0)
    endid = last_index_of_synapse_per_group[synapsegroupid];
  if ((synapsegroupid > 0) && (synapsegroupid < last_index_of_synapse_per_group.size())){
    startid = last_index_of_synapse_per_group[synapsegroupid - 1];
  }
  Synapses::save_connectivity_as_binary(path, prefix, synapsegroupid);
  std::ofstream delayfile;

  // Open output files
  delayfile.open((path + "/" + prefix + "SynapticDelays.bin"), std::ios::out | std::ios::binary);

  // Ensure weight data has been copied to frontend
  if (_backend)
    backend()->copy_to_frontend();

  // Send data to file
  for (int i = startid; i < endid; i++){
    delayfile.write((char *)&delays[synapse_reversesort_indices[i]], sizeof(int));
  }

  // Close files
  delayfile.close();
}

SPIKE_MAKE_INIT_BACKEND(SpikingSynapses);
