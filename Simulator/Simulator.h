#ifndef Simulator_H
#define Simulator_H
// Silences the printfs
//#define QUIETSTART

// cuRand Library import
#include <curand.h>
#include <curand_kernel.h>

//	CUDA library
#include <cuda.h>

#include "../RecordingElectrodes/CountNeuronSpikesRecordingElectrodes.h"
#include "../RecordingElectrodes/CollectNeuronSpikesRecordingElectrodes.h"
#include "../RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Models/SpikingModel.h"


struct Simulator_Recording_Electrodes_To_Use_Struct {

	Simulator_Recording_Electrodes_To_Use_Struct(): count_neuron_spikes_recording_electrodes_bool(false), input_count_neuron_spikes_recording_electrodes_bool(false), collect_neuron_spikes_recording_electrodes_bool(false), input_collect_neuron_spikes_recording_electrodes_bool(false), network_state_archive_recording_electrodes_bool(false)  {}

	bool count_neuron_spikes_recording_electrodes_bool;
	bool input_count_neuron_spikes_recording_electrodes_bool;
	bool collect_neuron_spikes_recording_electrodes_bool;
	bool input_collect_neuron_spikes_recording_electrodes_bool;
	bool network_state_archive_recording_electrodes_bool;

};


struct Simulator_File_Storage_Options_Struct {

	Simulator_File_Storage_Options_Struct(): save_recorded_neuron_spikes_to_file(false), save_recorded_input_neuron_spikes_to_file(false), write_initial_synaptic_weights_to_file_bool(false) {}

	bool save_recorded_neuron_spikes_to_file;
	bool save_recorded_input_neuron_spikes_to_file;
	bool write_initial_synaptic_weights_to_file_bool;

};


// Simulator Class for running of the simulations
class Simulator{
public:
	// Constructor/Destructor
	Simulator();
	~Simulator();


	// Variables
	float timestep;
	std::string full_directory_name_for_simulation_data_files;
	bool high_fidelity_spike_storage; // Flag: Enable for high accuracy spike storage, Disable for speed
	int number_of_simulations_run;

	// Host Pointers
	SpikingModel * spiking_model;
	Simulator_Recording_Electrodes_To_Use_Struct * recording_electrodes_to_use_struct;
	CountNeuronSpikesRecordingElectrodes* count_neuron_spikes_recording_electrodes;
	CountNeuronSpikesRecordingElectrodes* input_count_neuron_spikes_recording_electrodes;
	CollectNeuronSpikesRecordingElectrodes* collect_neuron_spikes_recording_electrodes;
	CollectNeuronSpikesRecordingElectrodes* input_collect_neuron_spikes_recording_electrodes;
	NetworkStateArchiveRecordingElectrodes* network_state_archive_recording_electrodes;
	
	// Functions
	void SetTimestep(float timest);
	void SetSpikingModel(SpikingModel * spiking_model_parameter);

	void CreateDirectoryForSimulationDataFiles(std::string directory_name_for_simulation_data_files);
	
	void prepare_recording_electrodes(Simulator_Recording_Electrodes_To_Use_Struct * recording_electrodes_to_use_struct);
	void reset_all_recording_electrodes();

	// void RunSimulationToCountNeuronSpikes(float presentation_time_per_stimulus_per_epoch, bool collect_spikes, bool save_collected_spikes_and_states_to_file, SpikeAnalyser *spike_analyser, bool human_readable_storage, bool isTrained);
	// void RunSimulationToCollectEvents(float presentation_time_per_stimulus_per_epoch, bool isTrained);
	// void RunSimulationToTrainNetwork(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, Stimuli_Presentation_Struct * stimuli_presentation_params, int stimulus_presentation_order_seed);
	void RunSimulation(Simulator_File_Storage_Options_Struct simulator_file_storage_options_struct, float presentation_time_per_stimulus_per_epoch, int number_of_epochs, bool save_collected_spikes_and_states_to_file, bool apply_stdp_to_relevant_synapses, bool count_spikes_per_neuron_for_single_cell_analysis, Stimuli_Presentation_Struct * stimuli_presentation_params, int stimulus_presentation_order_seed, SpikeAnalyser *spike_analyser, bool human_readable_storage, bool isTrained);


protected: 
	void perform_per_timestep_recording_electrode_instructions(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_stimulus_per_epoch);
	void perform_pre_stimulus_presentation_instructions();
	void perform_post_stimulus_presentation_instructions();
	void perform_post_epoch_instructions();
	void perform_end_of_simulation_instructions();
};
#endif
