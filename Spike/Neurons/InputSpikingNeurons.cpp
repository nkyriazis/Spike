#include "InputSpikingNeurons.hpp"
#include <stdlib.h>
#include <algorithm>
#include "../Helpers/TerminalHelpers.hpp"

void InputSpikingNeurons::select_stimulus(int stimulus_index){
  if (stimulus_index < total_number_of_input_stimuli){
    printf("Selecting Stimulus: %d\n", stimulus_index);
    current_stimulus_index = stimulus_index;
  } else {
    print_message_and_exit("Stimulus number exceeds number of stimuli loaded\n");
  }
}


