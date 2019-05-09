//  Synapse Class C++
//  Synapse.cpp
//
//  Author: Nasir Ahmad
//  Date: 7/12/2015

#include "Synapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

#include <algorithm> // for random shuffle
#include <vector> // for random shuffle
#include <random>

// Synapses Constructor
Synapses::Synapses() : Synapses(42) {
}

// Synapses Constructor
Synapses::Synapses(int seedval) {
  srand(seedval); // Seeding the random numbers
  random_state_manager = new RandomStateManager();
}

void Synapses::prepare_backend_early() {
  random_state_manager->init_backend(backend()->context);
}

void Synapses::sort_synapses(){
  if (!synapses_sorted){
    // Get the unique set of pre-synaptic neurons
    for (int pre=0; pre < pre_neuron_pointers.size(); pre++){
      bool found = false;
      for (int u=0; u < unique_pre_neuron_set.size(); u++){
        if (unique_pre_neuron_set[u] == pre_neuron_pointers[pre]){
          found = true;
          break;
        }
      }

      if (!found){
        unique_pre_neuron_set.push_back(pre_neuron_pointers[pre]);
       }
    }

    // Organise all synapses by pre-synaptic neuron, regardless of set
    int num_sorted_synapses = 0;
    for (int p = 0; p < unique_pre_neuron_set.size(); p++){
      // For each pre-synaptic neuron, collect its efferent synapses
      int total_pre_neurons = unique_pre_neuron_set[p]->total_number_of_neurons;
      std::vector<std::vector<int>> per_pre_neuron_synapses;
      for (int n=0; n < total_pre_neurons; n++){
        std::vector<int> empty;
        per_pre_neuron_synapses.push_back(empty);
      }

      for (int s=0; s < total_number_of_synapses; s++){
        if (pre_neuron_set[synapse_neuron_group_indices[s]] == unique_pre_neuron_set[p]){
          synapse_set_indices.push_back(p);
          per_pre_neuron_synapses[presynaptic_neuron_indices[s]].push_back(s);
        }
      }
     
      std::vector<int> efferent_num_per_pre; 
      std::vector<int> efferent_start_per_pre; 
      maximum_number_of_efferent_synapses_per_group.push_back(0);
      for (int n=0; n < total_pre_neurons; n++){
        efferent_num_per_pre.push_back((int)per_pre_neuron_synapses[n].size());
        efferent_start_per_pre.push_back(num_sorted_synapses);
        if (efferent_num_per_pre[n] > maximum_number_of_efferent_synapses_per_set[p])
          maximum_number_of_efferent_synapses_per_set[p] = efferent_num_per_pre[n];
        for (int s=0; s < per_pre_neuron_synapses[n].size(); s++){
          synapse_sort_indices[num_sorted_synapses] = per_pre_neuron_synapses[n][s];
          num_sorted_synapses++;
        }
      }

      efferent_num_per_set.push_back(efferent_num_per_pre);
      efferent_starts_per_set.push_back(efferent_start_per_pre);
    }

    if (num_sorted_synapses != total_number_of_synapses)
      print_message_and_exit("Issue encountered when sorting synapses: incomplete sort.");
    
    
    int* temp_presyn_array = (int*)malloc(total_number_of_synapses * sizeof(int));
    int* temp_postsyn_array = (int*)malloc(total_number_of_synapses * sizeof(int));
    float* temp_weight_array = (float*)malloc(total_number_of_synapses * sizeof(float));
    float* temp_scaling_array = (float*)malloc(total_number_of_synapses * sizeof(float));
    int* temp_synapse_neuron_group_indices = (int*)malloc(total_number_of_synapses * sizeof(int));
    // Re-ordering arrays
    for (int s=0; s < total_number_of_synapses; s++){
      temp_presyn_array[s] = presynaptic_neuron_indices[synapse_sort_indices[s]];
      temp_postsyn_array[s] = postsynaptic_neuron_indices[synapse_sort_indices[s]];
      temp_weight_array[s] = synaptic_efficacies_or_weights[synapse_sort_indices[s]];
      temp_scaling_array[s] = weight_scaling_constants[synapse_sort_indices[s]];
      temp_synapse_neuron_group_indices[s] = synapse_neuron_group_indices[synapse_sort_indices[s]];

      synapse_reversesort_indices[synapse_sort_indices[s]] = s;
    }

    free(presynaptic_neuron_indices);
    free(postsynaptic_neuron_indices);
    free(synaptic_efficacies_or_weights);
    free(weight_scaling_constants);
    free(synapse_neuron_group_indices);

    if (temp_presyn_array) presynaptic_neuron_indices = temp_presyn_array;
    if (temp_postsyn_array) postsynaptic_neuron_indices = temp_postsyn_array;
    if (temp_weight_array) synaptic_efficacies_or_weights = temp_weight_array;
    if (temp_scaling_array) weight_scaling_constants = temp_scaling_array;
    if (temp_synapse_neuron_group_indices) synapse_neuron_group_indices = temp_synapse_neuron_group_indices;

    synapses_sorted = true;
  }
}

// Synapses Destructor
Synapses::~Synapses() {
  free(presynaptic_neuron_indices);
  free(postsynaptic_neuron_indices);
  free(synaptic_efficacies_or_weights);
  free(synapse_sort_indices);
  free(synapse_reversesort_indices);
  free(weight_scaling_constants);
  free(synapse_neuron_group_indices);

  delete random_state_manager;
}


void Synapses::reset_state() {
  backend()->reset_state();
}


int Synapses::AddGroup(int presynaptic_group_id, 
                       int postsynaptic_group_id, 
                       Neurons * pre_neurons,
                       Neurons * post_neurons,
                       float timestep,
                       synapse_parameters_struct * synapse_params) {

  if (print_synapse_group_details == true) {
          printf("Adding synapse group...\n");
          printf("presynaptic_group_id: %d\n", presynaptic_group_id);
          printf("postsynaptic_group_id: %d\n", postsynaptic_group_id);
  }

  // Determine which neuron set for this synapse population
  pre_neuron_pointers.push_back(pre_neurons);
  post_neuron_pointers.push_back(post_neurons);
  int synapse_set_group = pre_neuron_pointers.size() - 1;


  // Getting group shapes, and pre and post-synaptic neuron indices
  int * presynaptic_group_shape = pre_neurons->group_shapes[presynaptic_group_id];
  int * postsynaptic_group_shape = post_neurons->group_shapes[postsynaptic_group_id];

  int prestart = pre_neurons->start_neuron_indices_for_each_group[presynaptic_group_id];
  int preend = pre_neurons->last_neuron_indices_for_each_group[presynaptic_group_id];
  int poststart = post_neurons->start_neuron_indices_for_each_group[postsynaptic_group_id];
  int postend = post_neurons->last_neuron_indices_for_each_group[postsynaptic_group_id];


  if (print_synapse_group_details == true) {
          printf("Presynaptic neurons start index: %d (neurons)\n", prestart);
          printf("Presynaptic neurons end index: %d (neurons)\n", preend);
          printf("Postsynaptic neurons start index: %d (neurons)\n", poststart);
          printf("Postsynaptic neurons end index: %d (neurons)\n", postend);
  }


  int original_number_of_synapses = total_number_of_synapses;

  // Carry out the creation of the connectivity matrix
  switch (synapse_params->connectivity_type){
            
        case CONNECTIVITY_TYPE_ALL_TO_ALL:
          {
            
            int increment = (preend-prestart)*(postend-poststart);
            Synapses::increment_number_of_synapses(increment, synapse_group_id);

            // If the connectivity is all_to_all
            for (int i = prestart; i < preend; i++){
              for (int j = poststart; j < postend; j++){
                // Index
                int idx = original_number_of_synapses + (i-prestart)*(postend-poststart) + (j-poststart);
                // Setup Synapses
                presynaptic_neuron_indices[idx] = i;
                postsynaptic_neuron_indices[idx] = j;
              }
            }
            break;
          }
        case CONNECTIVITY_TYPE_ONE_TO_ONE:
          {
            if ((preend-prestart) != (postend-poststart)) print_message_and_exit("Unequal populations for one_to_one.");
            
            int increment = (preend-prestart);
            Synapses::increment_number_of_synapses(increment, synapse_group_id);
            
            // If the connectivity is one_to_one
            // Create the connectivity
            for (int i = 0; i < (preend-prestart); i++){
              presynaptic_neuron_indices[original_number_of_synapses + i] = prestart + i;
              postsynaptic_neuron_indices[original_number_of_synapses + i] = poststart + i;
            }

            break;
          }
        case CONNECTIVITY_TYPE_RANDOM:
          {
            // If the connectivity is random
            // Begin a count
            for (int i = prestart; i < preend; i++){
              for (int j = poststart; j < postend; j++){
                // Probability of connection
                float prob = ((float)rand() / (RAND_MAX));
                // If it is within the probability range, connect!
                if (prob < synapse_params->random_connectivity_probability){
            
                  Synapses::increment_number_of_synapses(1, synapse_group_id);
                  presynaptic_neuron_indices[total_number_of_synapses - 1] = i;
                  postsynaptic_neuron_indices[total_number_of_synapses - 1] = j;
                }
              }
            }
            break;
          }
    
        case CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE:
          {

            float standard_deviation_sigma = synapse_params->gaussian_synapses_standard_deviation;
            int number_of_new_synapses_per_postsynaptic_neuron = synapse_params->gaussian_synapses_per_postsynaptic_neuron;
            int number_of_presynaptic_neurons = presynaptic_group_shape[0] * presynaptic_group_shape[1];
            if (number_of_new_synapses_per_postsynaptic_neuron > number_of_presynaptic_neurons){
              print_message_and_exit("Synapse creation error. Pre-synaptic population smaller than requested synapses per post-synaptic neuron (Gaussian Sampling).");
            }
      
            int number_of_postsynaptic_neurons_in_group = postend - poststart;
            int total_number_of_new_synapses = number_of_new_synapses_per_postsynaptic_neuron * number_of_postsynaptic_neurons_in_group;
            Synapses::increment_number_of_synapses(total_number_of_new_synapses, synapse_group_id);

            for (int postid = 0; postid < number_of_postsynaptic_neurons_in_group; postid++){
              float post_fractional_centre_x = ((float)(postid % postsynaptic_group_shape[0]) / (float)postsynaptic_group_shape[0]);
              float post_fractional_centre_y = ((float)((float)postid / (float)postsynaptic_group_shape[1]) / (float)postsynaptic_group_shape[1]);
              int pre_centre_x = presynaptic_group_shape[0] * post_fractional_centre_x;
              int pre_centre_y = presynaptic_group_shape[1] * post_fractional_centre_y;

              // Constructing the probability with which we should connect to each pre-synaptic neuron
              std::vector<float> pre_neuron_probabilities;
              std::vector<float> pre_neuron_indices;
              float total_probability = 0.0;
              for (int preid = 0; preid < number_of_presynaptic_neurons; preid++){
                int pre_xcoord = preid % presynaptic_group_shape[0]; 
                int pre_ycoord = preid / presynaptic_group_shape[0];
                float probability_of_connection = expf(- (powf((pre_xcoord - pre_centre_x), 2.0)) / (2.0 * powf(standard_deviation_sigma, 2)));
                probability_of_connection *= expf(- (powf((pre_ycoord - pre_centre_y), 2.0)) / (2.0 * powf(standard_deviation_sigma, 2)));
                pre_neuron_probabilities.push_back(probability_of_connection);
                pre_neuron_indices.push_back(preid);
                total_probability += probability_of_connection;
              }
              
              for (int i=0; i < number_of_new_synapses_per_postsynaptic_neuron; i++){
                postsynaptic_neuron_indices[original_number_of_synapses + postid*number_of_new_synapses_per_postsynaptic_neuron + i] = poststart + postid;
                float randval = total_probability*((float)rand() / (RAND_MAX));
                float probability_trace = 0.0;
                for (int preloc = 0; preloc < pre_neuron_probabilities.size(); preloc++){
                  if ((randval > probability_trace) && (randval < (probability_trace + pre_neuron_probabilities[preloc]))){
                    int chosenpreid = prestart + pre_neuron_indices[preloc];
                    presynaptic_neuron_indices[original_number_of_synapses + postid*number_of_new_synapses_per_postsynaptic_neuron + i] = chosenpreid;
                    total_probability -= pre_neuron_probabilities[preloc];
                    pre_neuron_indices.erase(pre_neuron_indices.begin() + preloc);
                    pre_neuron_probabilities.erase(pre_neuron_probabilities.begin() + preloc);
                    break;
                  } else {
                    probability_trace += pre_neuron_probabilities[preloc];
                  }
                }
              }
            }

            if (total_number_of_new_synapses > largest_synapse_group_size) {
              largest_synapse_group_size = total_number_of_new_synapses;
            }

            break;
          }
        case CONNECTIVITY_TYPE_PAIRWISE:
          {
      // Check that the number of pre and post syns are equivalent
      if (synapse_params->pairwise_connect_presynaptic.size() != synapse_params->pairwise_connect_postsynaptic.size()){
        std::cerr << "Synapse pre and post vectors are not the same length!" << std::endl;
        exit(1);
      }

      Synapses::increment_number_of_synapses(synapse_params->pairwise_connect_presynaptic.size(), synapse_group_id);

      int numpostneurons = postsynaptic_group_shape[0]*postsynaptic_group_shape[1];
      int numpreneurons = presynaptic_group_shape[0]*presynaptic_group_shape[1];
      for (int i=0; i < synapse_params->pairwise_connect_presynaptic.size(); i++){
        if ((synapse_params->pairwise_connect_presynaptic[i] < 0) || (synapse_params->pairwise_connect_postsynaptic[i] < 0)){
          print_message_and_exit("PAIRWISE CONNECTION ERROR: Negative pre/post indices encountered. All indices should be positive (relative to the number of neurons in the pre/post synaptic neuron groups).");
        }
        if ((synapse_params->pairwise_connect_presynaptic[i] >= numpreneurons) || (synapse_params->pairwise_connect_postsynaptic[i] >= numpostneurons)){
          print_message_and_exit("PAIRWISE CONNECTION ERROR: Pre/post indices encountered too large. All indices should be up to the size of the neuron group. Indexing is from zero.");
        }
        presynaptic_neuron_indices[original_number_of_synapses + i] = prestart + int(synapse_params->pairwise_connect_presynaptic[i]);
        postsynaptic_neuron_indices[original_number_of_synapses + i] = poststart + int(synapse_params->pairwise_connect_postsynaptic[i]);
      }

      break;
    }
    default:
    {
      print_message_and_exit("Unknown Connection Type.");
      break;
    }
  }



  temp_number_of_synapses_in_last_group = total_number_of_synapses - original_number_of_synapses;

  // Setting up weights and defining neuron efferent/afferent numbers
  for (int i = original_number_of_synapses; i < total_number_of_synapses; i++){
    weight_scaling_constants[i] = synapse_params->weight_scaling_constant;
    synaptic_efficacies_or_weights[i] = synapse_params->weight_range[0] + (synapse_params->weight_range[1] - synapse_params->weight_range[0])*((float)rand() / (RAND_MAX));
    synapse_neuron_group_indices[i] = synapse_group_id;
    
    if (synapse_params->connectivity_type == CONNECTIVITY_TYPE_PAIRWISE){
      if (synapse_params->pairwise_connect_weight.size() == temp_number_of_synapses_in_last_group){
        synaptic_efficacies_or_weights[i] = synapse_params->pairwise_connect_weight[i - original_number_of_synapses];
      } else if (synapse_params->pairwise_connect_weight.size() != 0) {
        print_message_and_exit("PAIRWISE CONNECTION ISSUE: Weight vector length not as expected. Should be the same length as pre/post vecs.");
      }
    }

    // Used for event count
    post_neurons->per_neuron_afferent_synapse_count[postsynaptic_neuron_indices[i]]++;

    //if (post_neurons->per_neuron_afferent_synapse_count[postsynaptic_neuron_indices[i]] > maximum_number_of_afferent_synapses)
    //  maximum_number_of_afferent_synapses = post_neurons->per_neuron_afferent_synapse_count[postsynaptic_neuron_indices[i]];

    int presynaptic_id = presynaptic_neuron_indices[i];
    pre_neurons->AddEfferentSynapse(presynaptic_id, i);

    synapse_sort_indices[i] = i;
    synapse_reversesort_indices[i] = i;
  }

  // Set up plasticity
  int plasticity_id = -1;
  int original_num_plasticity_indices = 0;
  if (synapse_params->plasticity_vec.size() > 0){
    for (int vecid = 0; vecid < synapse_params->plasticity_vec.size(); vecid++){
      Plasticity* plasticity_ptr = synapse_params->plasticity_vec[vecid];
      //Check first for nullptr
      if (plasticity_ptr == nullptr)
        continue;
      plasticity_id = plasticity_ptr->plasticity_rule_id;
      // Provide plasticity relevant synapse indices
      plasticity_ptr->AddSynapseIndices((total_number_of_synapses - temp_number_of_synapses_in_last_group), temp_number_of_synapses_in_last_group);
    }
  }


  post_start_per_group.push_back(poststart);
  pre_start_per_group.push_back(prestart);
  last_index_of_synapse_per_group.push_back(total_number_of_synapses);

  // Returning the index of synapses
  return(last_index_of_synapse_per_group.size() - 1);

}


void Synapses::increment_number_of_synapses(int increment, int synapse_group_id) {

  total_number_of_synapses += increment;

  if (total_number_of_synapses - increment == 0) {
          presynaptic_neuron_indices = (int*)malloc(total_number_of_synapses * sizeof(int));
          postsynaptic_neuron_indices = (int*)malloc(total_number_of_synapses * sizeof(int));
          synaptic_efficacies_or_weights = (float*)malloc(total_number_of_synapses * sizeof(float));
          weight_scaling_constants = (float*)malloc(total_number_of_synapses * sizeof(float));
          synapse_neuron_group_indices = (int*)malloc(total_number_of_synapses * sizeof(int));
          synapse_sort_indices = (int*)malloc(total_number_of_synapses * sizeof(int));
          synapse_reversesort_indices = (int*)malloc(total_number_of_synapses * sizeof(int));
  } else {
    int* temp_presynaptic_neuron_indices = (int*)realloc(presynaptic_neuron_indices, total_number_of_synapses * sizeof(int));
    int* temp_postsynaptic_neuron_indices = (int*)realloc(postsynaptic_neuron_indices, total_number_of_synapses * sizeof(int));
    float* temp_synaptic_efficacies_or_weights = (float*)realloc(synaptic_efficacies_or_weights, total_number_of_synapses * sizeof(float));
    float* temp_weight_scaling_constants = (float*)realloc(weight_scaling_constants, total_number_of_synapses * sizeof(float));
    int* temp_synapse_neuron_group_indices = (int*)realloc(synapse_neuron_group_indices, total_number_of_synapses * sizeof(int));
    int* temp_sort_indices = (int*)realloc(synapse_sort_indices, total_number_of_synapses * sizeof(int));
    int* temp_revsort_indices = (int*)realloc(synapse_reversesort_indices, total_number_of_synapses * sizeof(int));

    if (temp_presynaptic_neuron_indices != nullptr) presynaptic_neuron_indices = temp_presynaptic_neuron_indices;
    if (temp_postsynaptic_neuron_indices != nullptr) postsynaptic_neuron_indices = temp_postsynaptic_neuron_indices;
    if (temp_synaptic_efficacies_or_weights != nullptr) synaptic_efficacies_or_weights = temp_synaptic_efficacies_or_weights;
    if (temp_weight_scaling_constants != nullptr) weight_scaling_constants = temp_weight_scaling_constants;
    if (temp_synapse_neuron_group_indices != nullptr) synapse_neuron_group_indices = temp_synapse_neuron_group_indices;
    if (temp_sort_indices != nullptr) synapse_sort_indices = temp_sort_indices;
    if (temp_revsort_indices != nullptr) synapse_reversesort_indices = temp_revsort_indices;
  }

}

void Synapses::save_connectivity_as_txt(std::string path, std::string prefix, int synapsegroupid){
  if (synapsegroupid < 0)
    print_message_and_exit("Synapse saving error: Provide a non-zero synapse group id!\n");
  int startid = 0;
  int endid = total_number_of_synapses;
  if (synapsegroupid >= 0)
    endid = last_index_of_synapse_per_group[synapsegroupid];
  if ((synapsegroupid > 0) && (synapsegroupid < last_index_of_synapse_per_group.size())){
    startid = last_index_of_synapse_per_group[synapsegroupid - 1];
  }
  int precorrection = 0;
  int postcorrection = 0;
  if (synapsegroupid >= 0){
    postcorrection = post_start_per_group[synapsegroupid];
    precorrection = pre_start_per_group[synapsegroupid];
  }
  std::ofstream preidfile, postidfile, weightfile;

  // Open output files
  preidfile.open((path + "/" + prefix + "PresynapticIDs.txt"), std::ios::out | std::ios::binary);
  postidfile.open((path + "/" + prefix + "PostsynapticIDs.txt"), std::ios::out | std::ios::binary);
  weightfile.open((path + "/" + prefix + "SynapticWeights.txt"), std::ios::out | std::ios::binary);

  // Ensure weight data has been copied to frontend
  if (_backend)
    backend()->copy_to_frontend();

  // Send data to file
  for (int i = startid; i < endid; i++){
    preidfile << presynaptic_neuron_indices[synapse_reversesort_indices[i]] - precorrection << std::endl;
    postidfile << postsynaptic_neuron_indices[synapse_reversesort_indices[i]] - postcorrection << std::endl;
    weightfile << synaptic_efficacies_or_weights[synapse_reversesort_indices[i]] << std::endl;
  }

  // Close files
  preidfile.close();
  postidfile.close();
  weightfile.close();

};
// Ensure copied from device, then send
void Synapses::save_connectivity_as_binary(std::string path, std::string prefix, int synapsegroupid){
  if (synapsegroupid < 0)
    print_message_and_exit("Synapse saving error: Provide a non-zero synapse group id!\n");
  int startid = 0;
  int endid = total_number_of_synapses;
  if (synapsegroupid >= 0)
    endid = last_index_of_synapse_per_group[synapsegroupid];
  if ((synapsegroupid > 0) && (synapsegroupid < last_index_of_synapse_per_group.size())){
    startid = last_index_of_synapse_per_group[synapsegroupid - 1];
  }
  int precorrection = 0;
  int postcorrection = 0;
  if (synapsegroupid >= 0){
    postcorrection = post_start_per_group[synapsegroupid];
    precorrection = pre_start_per_group[synapsegroupid];
  }
  std::ofstream preidfile, postidfile, weightfile;

  // Open output files
  preidfile.open((path + "/" + prefix + "PresynapticIDs.bin"), std::ios::out | std::ios::binary);
  postidfile.open((path + "/" + prefix + "PostsynapticIDs.bin"), std::ios::out | std::ios::binary);
  weightfile.open((path + "/" + prefix + "SynapticWeights.bin"), std::ios::out | std::ios::binary);

  // Ensure weight data has been copied to frontend
  if (_backend)
    backend()->copy_to_frontend();

  // Send data to file
  int preid, postid;
  float weight;
  for (int i = startid; i < endid; i++){
    preid = presynaptic_neuron_indices[synapse_reversesort_indices[i]] - precorrection;
    postid = postsynaptic_neuron_indices[synapse_reversesort_indices[i]] - postcorrection;
    weight = synaptic_efficacies_or_weights[synapse_reversesort_indices[i]];
    preidfile.write((char *)&preid, sizeof(int));
    postidfile.write((char *)&postid, sizeof(int));
    weightfile.write((char *)&weight, sizeof(float));
  }
  /*
  // Send data to file
  preidfile.write((char *)&presynaptic_neuron_indices[startid], (endid - startid)*sizeof(int));
  postidfile.write((char *)&postsynaptic_neuron_indices[startid], (endid - startid)*sizeof(int));
  */
  //weightfile.write((char *)&synaptic_efficacies_or_weights[startid], (endid - startid)*sizeof(float));

  // Close files
  preidfile.close();
  postidfile.close();
  weightfile.close();
}

// Load Network??
//void Synapses::load_connectivity_from_txt(std::string path, std::string prefix);
//void Synapses::load_connectivity_from_binary(std::string path, std::string prefix);

void Synapses::save_weights_as_txt(std::string path, std::string prefix, int synapsegroupid){
  if (synapsegroupid < 0)
    print_message_and_exit("Synapse saving error: Provide a non-zero synapse group id!\n");
  int startid = 0;
  int endid = total_number_of_synapses;
  if (synapsegroupid >= 0)
    endid = last_index_of_synapse_per_group[synapsegroupid];
  if ((synapsegroupid > 0) && (synapsegroupid < last_index_of_synapse_per_group.size())){
    startid = last_index_of_synapse_per_group[synapsegroupid - 1];
  }

  std::ofstream weightfile;
  if (_backend)
    backend()->copy_to_frontend();
  weightfile.open((path + "/" + prefix + "SynapticWeights.txt"), std::ios::out | std::ios::binary);
  for (int i = startid; i < endid; i++){
    weightfile << synaptic_efficacies_or_weights[synapse_reversesort_indices[i]] << std::endl;
  }
  weightfile.close();
}

void Synapses::save_weights_as_binary(std::string path, std::string prefix, int synapsegroupid){
  if (synapsegroupid < 0)
    print_message_and_exit("Synapse saving error: Provide a non-zero synapse group id!\n");
  int startid = 0;
  int endid = total_number_of_synapses;
  if (synapsegroupid >= 0)
    endid = last_index_of_synapse_per_group[synapsegroupid];
  if ((synapsegroupid > 0) && (synapsegroupid < last_index_of_synapse_per_group.size())){
    startid = last_index_of_synapse_per_group[synapsegroupid - 1];
  }

  std::ofstream weightfile;
  if (_backend)
    backend()->copy_to_frontend();
  weightfile.open((path + "/" + prefix + "SynapticWeights.bin"), std::ios::out | std::ios::binary);
  for (int i = startid; i < endid; i++){
    weightfile.write((char *)&synaptic_efficacies_or_weights[synapse_reversesort_indices[i]], sizeof(float));
  }
  
  weightfile.close();
}


void Synapses::load_weights(std::vector<float> weights, int synapsegroupid){
  if (synapsegroupid < 0)
    print_message_and_exit("Synapse saving error: Provide a non-zero synapse group id!\n");
  int startid = 0;
  int endid = total_number_of_synapses;
  if (synapsegroupid >= 0)
    endid = last_index_of_synapse_per_group[synapsegroupid];
  if ((synapsegroupid > 0) && (synapsegroupid < last_index_of_synapse_per_group.size())){
    startid = last_index_of_synapse_per_group[synapsegroupid - 1];
  }

  if (weights.size() == (endid - startid)){
    for (int i = startid; i < endid; i++){
      synaptic_efficacies_or_weights[synapse_reversesort_indices[i]] = weights[i - startid];
    }
  } else {
    print_message_and_exit("Number of weights loading not equal to number of synapses!!");
  }
  
  if (_backend)
    backend()->copy_to_backend();
}

void Synapses::load_weights_from_txt(std::string filepath, int synapsegroupid){
  std::ifstream weightfile;
  weightfile.open(filepath, std::ios::in | std::ios::binary);
  
  // Getting values into a vector
  std::vector<float> loadingweights;
  float weightval = 0.0f;
  while (weightfile >> weightval){
    loadingweights.push_back(weightval);
  }
  weightfile.close();

  load_weights(loadingweights, synapsegroupid);

}
void Synapses::load_weights_from_binary(std::string filepath, int synapsegroupid){
  std::ifstream weightfile;
  weightfile.open(filepath, std::ios::in | std::ios::binary);

  // Getting values into a vector
  std::vector<float> loadingweights;
  float weightval = 0.0f;
  while (weightfile.read(reinterpret_cast<char*>(&weightval), sizeof(float))){
    loadingweights.push_back(weightval);
  }
  weightfile.close();
  
  load_weights(loadingweights, synapsegroupid);
}


SPIKE_MAKE_STUB_INIT_BACKEND(Synapses);
