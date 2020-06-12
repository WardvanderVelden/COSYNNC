#include "Procedure.h"

namespace COSYNNC {
	// Default constructor
	Procedure::Procedure() { 
		_stateQuantizer = nullptr;
		_inputQuantizer = nullptr;

		_plant = nullptr;

		std::cout << "COSYNNC:\tA correct-by-design neural network controller synthesis procedure" << std::endl;
	}


	// Default destructor
	Procedure::~Procedure() {
		delete _abstraction;

		delete _stateQuantizer;
		delete _inputQuantizer;

		delete _plant;

		delete _verifier;
	}


	#pragma region Procedure Specifiers

	// Set the state quantizer to the specified quantization parameters
	void Procedure::SpecifyStateQuantizer(Vector eta, Vector lowerBound, Vector upperBound) {
		_stateQuantizer = new Quantizer();
		_stateQuantizer->SetQuantizeParameters(eta, lowerBound, upperBound);

		_singleStateTrainingFocus = Vector(_stateQuantizer->GetDimension());

		Log("COSYNNC", "State quantizer specified");
	}


	// Set the input quantizer to the specified quantization parameters
	void Procedure::SpecifyInputQuantizer(Vector eta, Vector lowerBound, Vector upperBound) {
		_inputQuantizer = new Quantizer();
		_inputQuantizer->SetQuantizeParameters(eta, lowerBound, upperBound);

		Log("COSYNNC", "Input quantizer specified");
	}


	// Specify if the network should reinforce upon reaching the winning set (only applicable to reachability)
	void Procedure::SpecifyWinningSetReinforcement(bool reinforce) {
		// TEMPORARY: Winning set reinforcement should also work for non reachability specifications?
		//if (_specification.GetSpecificationType() == ControlSpecificationType::Reachability) {
			_useWinningSetReinforcement = reinforce;
		//}

		if (_useWinningSetReinforcement) {
			Log("COSYNNC", "Network will reinforce upon reaching the winning set");
		}
		else {
			Log("COSYNNC", "Network will not reinforce upon reaching the winning set");
		}
	}


	// Set the synthesis parameters
	void Procedure::SpecifySynthesisParameters(unsigned int maxEpisodes, unsigned int maxEpisodeHorizonTrainer, unsigned int verboseEpisode, unsigned int verificationEpisode, unsigned int maxEpisodeHorizonVerifier) {
		_maxEpisodes = maxEpisodes;

		_verboseEpisode = verboseEpisode;
		_verificationEpisode = (verificationEpisode == 0) ? _verboseEpisode * 10 : verificationEpisode;

		_maxEpisodeHorizonTrainer = maxEpisodeHorizonTrainer;
		_maxEpisodeHorizonVerifier = (maxEpisodeHorizonVerifier == 0) ? _maxEpisodeHorizonTrainer * 3 : maxEpisodeHorizonVerifier;

		Log("COSYNNC", "Synthesis parameters specified");
	}


	// Set the control specification
	void Procedure::SpecifyControlSpecification(ControlSpecificationType type, Vector lowerBound, Vector upperBound) {
		_specification = ControlSpecification(type, _plant);
		_specification.SetHyperInterval(lowerBound, upperBound);

		_controller.SetControlSpecification(&_specification);

		Log("COSYNNC", "Control specification defined");
	}


	// Specify the radial initial state selection based on the synthesis progression
	void Procedure::SpecifyRadialInitialState(float lower, float upper) {
		_radialInitialStateAvailable = true;

		_radialInitialStateLower = lower;
		_radialInitialStateUpper = (upper == 0.0) ? (1.0 - lower) : upper;

		Log("COSYNNC", "Radial from goal initial state selection specified");
	}


	// Specify the norm to be used during training
	void Procedure::SpecifyNorm(vector<float> normWeights) {
		_useNorm = true;

		_normWeights = normWeights;

		Log("COSYNNC", "Norm based reinforcement specified");
	}


	// Specify the verbosity of the procedure
	void Procedure::SpecifyVerbosity(bool verboseTrainer, bool verboseVerifier) {
		_verboseTrainer = verboseTrainer;
		if (_verboseTrainer) Log("COSYNNC", "Trainer set to verbose");
		else Log("COSYNNC", "Trainer set to non-verbose");

		_verboseVerifier = verboseVerifier;
		if (_verboseVerifier) Log("COSYNNC", "Verifier set to verbose");
		else Log("COSYNNC", "Verifier set to non-verbose");
	}


	// Specify the training focus that should be used during training
	void Procedure::SpecifyTrainingFocus(TrainingFocus trainingFocus, Vector singleStateTrainingFocus) {
		_trainingFocuses.push_back(trainingFocus);

		switch (trainingFocus) {
		case TrainingFocus::SingleState: Log("COSYNNC", "Single state training focus added"); break;
			case TrainingFocus::AllStates: Log("COSYNNC", "All states training focus added"); break;
			case TrainingFocus::RadialOutwards: Log("COSYNNC", "Radial outwards training focus added"); break;
			case TrainingFocus::LosingStates: Log("COSYNNC", "Losing domain training focus added"); break;
			case TrainingFocus::NeighboringLosingStates: Log("COSYNNC", "Losing states neighboring the winning states training focus added"); break;
		}
		
		if (singleStateTrainingFocus.GetLength() != 0) _singleStateTrainingFocus = singleStateTrainingFocus;
	}


	// Specify if the raw neural network should be saved
	void Procedure::SpecifySaveRawNeuralNetwork(bool save) {
		_saveRawNeuralNetwork = save;
	}


	// Specify if the apparent winning set should be computed
	void Procedure::SpecifyComputeApparentWinningSet(bool compute) {
		_computeApparentWinningSet = compute;
	}


	// Specify what type of transition calculation should be used
	void Procedure::SpecifyUseRefinedTransitions(bool rough) {
		_useRefinedTransitions = rough;

		if(rough) Log("COSYNNC", "Abstraction set to use rough transitions");
	}


	// Specify whether or not to save the transitions
	void Procedure::SpecifySaveAbstractionTransitions(bool saveTransitions) {
		_saveTransitions = saveTransitions;

		if (saveTransitions) Log("COSYNNC", "Abstraction transitions are being saved to increase computation speed");
		else Log("COSYNNC", "Abstraction transitions are not being saved");
	}


	// Set the plant
	void Procedure::SetPlant(Plant* plant) {
		_plant = plant;

		Log("COSYNNC", "Plant linked");
	}


	// Set the neural network
	void Procedure::SetNeuralNetwork(NeuralNetwork* neuralNetwork, size_t batchSize) {
		_neuralNetwork = neuralNetwork;
		//_neuralNetwork->ConfigurateInputOutput(_plant, _inputQuantizer, _maxEpisodeHorizonTrainer, 1.0);
		_neuralNetwork->ConfigurateInputOutput(_plant, _inputQuantizer, batchSize, 1.0);

		_outputType = _neuralNetwork->GetOutputType();

		Log("COSYNNC", "Neural network linked");

		auto dataSize = _neuralNetwork->GetDataSize();
		Log("COSYNNC", "Neural network has data size: " + std::to_string(dataSize) + " bytes");
	}

	#pragma endregion Procedure Specifiers	


	// Initialize the procedure, returns false if not all required parameters are specified
	bool Procedure::Initialize() {
		// Initialize controller
		_controller = Controller(_plant, _stateQuantizer, _inputQuantizer);
		_controller.SetNeuralNetwork(_neuralNetwork);
		_controller.SetControlSpecification(&_specification);

		// Initialize the abstraction
		_abstraction = new Abstraction(_plant, &_controller, _stateQuantizer, _inputQuantizer, &_specification);
		_abstraction->SetUseRefinedTransitions(_useRefinedTransitions);
		_abstraction->SetSaveTransitions(_saveTransitions);

		// Initialize verifier
		_verifier = new Verifier(_abstraction);
		_verifier->SetVerboseMode(_verboseVerifier);

		_fileManager = FileManager(_neuralNetwork, _verifier, _abstraction);
		_bddManager = BddManager(_abstraction);

		// TODO: Test if all required parameters are specified before the synthesis procedure begins

		_hasSuccesfullyInitialized = true;
		return true;
	}


	// Run the synthesis procedure
	void Procedure::Synthesize() {
		if (_hasSuccesfullyInitialized) {
			Log("COSYNNC", "Procedure was succesfully initialized, synthesize procedure started");

			Log("Trainer", "Starting training");
			for (int i = 0; i <= _maxEpisodes; i++) {
				IterateEpisode(i);

				if (i % _verificationEpisode == 0 && i != 0) {
					Verify(); 

					Log("File Manager", "Saving neural network with a timestamp.");
					SaveTimestampedNetwork();

					Log("COSYNNC", "Best controller so far is timestamp: " + _bestControllerTimestamp + " with winning domain percentage: " + to_string(_bestControllerWinningDomainPercentage) + "%");

					// TODO: Stop if verification is in specs
					Log("Trainer", "Continuing training.");
				}
			}
			Log("COSYNNC", "Synthesis terminated!");
		}
		else {
			Log("COSYNNC", "Procedure was not succesfully initialized, check if all required parameters are specified.");
		}

		MXNotifyShutdown();
	}


	// Iterate the training phase
	void Procedure::IterateEpisode(unsigned int episodeNumber) {
		if (episodeNumber % _verboseEpisode == 0 && _verboseTrainer) std::cout << std::endl;

		vector<Vector> states;
		vector<Vector> reinforcingLabels;
		vector<Vector> deterringLabels;

		// Get an initial state based on the control specification we are trying to solve for
		auto initialState = GetInitialStateForTraining(episodeNumber);
		_plant->SetState(initialState);

		// Define the norm for determining the networks performance
		double initialNorm = 0.0;
		if(_useNorm) initialNorm = (initialState - _specification.GetCenter()).GetWeightedNorm(_normWeights);
		float norm = initialNorm;

		bool isInSpecificationSet = false;
		bool isInWinningSet = false;

		bool stopEpisode = false;
		bool hasReachedSet = false;

		// Simulate the episode using the neural network
		for (int j = 0; j < _maxEpisodeHorizonTrainer; j++) {
			auto state = _plant->GetState();

			auto normalizedQuantizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->QuantizeVector(state));
			states.push_back(normalizedQuantizedState);

			// Define vectors to allow access in and outside the switch
			Vector input(_abstraction->GetInputQuantizer()->GetDimension());
			Vector networkOutput(_abstraction->GetInputQuantizer()->GetDimension());
			Vector newState(_abstraction->GetStateQuantizer()->GetDimension());

			// Get data for a single training step
			FormatTrainingData(state, &reinforcingLabels, &deterringLabels, &input, &networkOutput, &newState, &isInSpecificationSet, &norm);
			isInWinningSet = _verifier->IsIndexInWinningSet(_stateQuantizer->GetIndexFromVector(newState));

			// DEBUG: Print simulation for verification purposes
			if (episodeNumber % _verboseEpisode == 0 && _verboseTrainer) {
				auto verboseLabels = (_neuralNetwork->GetLabelDimension() <= 5) ? _neuralNetwork->GetLabelDimension() : 5;

				std::cout << "\ti: " << episodeNumber << "\tj: " << j << "\t";
				for (size_t i = 0; i < _abstraction->GetStateQuantizer()->GetDimension(); i++) std::cout << "\tx" << i << ": " << newState[i];
				for (size_t i = 0; i < verboseLabels; i++) {
					if (i == 0) std::cout << "\t";
					std::cout << "\tn" << i << ": " << networkOutput[i];;
				}

				auto verboseInputs = (_abstraction->GetInputQuantizer()->GetDimension() <= 5) ? _abstraction->GetInputQuantizer()->GetDimension() : 5;
				for (size_t i = 0; i < verboseInputs; i++) {
					if (i == 0) std::cout << "\t";
					std::cout << "\tu" << i << ": " << input[i];;
				}
				std::cout << "\ts: " << isInSpecificationSet << std::endl;
			}

			// Check episode stopping conditions
			switch (_specification.GetSpecificationType()) {
				case ControlSpecificationType::Invariance: if (!isInSpecificationSet) stopEpisode = true; break;
				case ControlSpecificationType::Reachability: if (isInSpecificationSet) stopEpisode = true; break;
				case ControlSpecificationType::ReachAndStay: 
					if (isInSpecificationSet && !hasReachedSet) hasReachedSet = true; 
					if (!isInSpecificationSet && hasReachedSet) stopEpisode = true;
					break;
			}

			if (_useWinningSetReinforcement && isInWinningSet) stopEpisode = true;

			if (!_stateQuantizer->IsInBounds(newState) || stopEpisode) break;
		}

		// Train the neural network based on the performance of the network
		if (norm < initialNorm && _useNorm) AddToTrainingQueue(states, reinforcingLabels);
		else if (isInSpecificationSet) AddToTrainingQueue(states, reinforcingLabels);
		else if (isInWinningSet && _useWinningSetReinforcement) AddToTrainingQueue(states, reinforcingLabels);
		else AddToTrainingQueue(states, deterringLabels);
	}


	// Formats the training data for a single training step
	void Procedure::FormatTrainingData(Vector state, vector<Vector>* reinforcingLabels, vector<Vector>* deterringLabels, Vector* input, Vector* networkOutput, Vector* newState, bool* isInSpecificationSet, float* norm) {
		switch (_outputType) {
			case OutputType::Labelled: {
				Vector oneHot(_inputQuantizer->GetCardinality());
				*input = _controller.GetProbabilisticControlActionFromLabelledNeurons(state, oneHot, *networkOutput);

				// Evolve the plant
				_plant->Evolve(*input);
				*newState = _plant->GetState();

				// See if the evolved state is in the specification set and calculate the new norm on the state
				*isInSpecificationSet = _specification.IsInSpecificationSet(*newState);
				if (_useNorm) *norm = (*newState - _specification.GetCenter()).GetWeightedNorm(_normWeights);

				// Create reinforcing and deterring labels
				Vector reinforcementLabel = Vector((int)_abstraction->GetInputQuantizer()->GetCardinality());
				Vector deterringLabel = Vector((int)_abstraction->GetInputQuantizer()->GetCardinality());

				reinforcementLabel = oneHot;

				float sum = 0.0;
				for (int i = 0; i < (int)_abstraction->GetInputQuantizer()->GetCardinality(); i++) {
					deterringLabel[i] = 1.0 - reinforcementLabel[i];
					sum += deterringLabel[i];
				}

				for (int i = 0; i < (int)_abstraction->GetInputQuantizer()->GetCardinality(); i++) deterringLabel[i] = deterringLabel[i] / sum;

				// Add labels to the list of labels
				reinforcingLabels->push_back(reinforcementLabel);
				deterringLabels->push_back(deterringLabel);

				break;
			}
			case OutputType::Range: {
				*input = _controller.GetProbabilisticControlActionFromRangeNeurons(state, *networkOutput);

				// Evolve the plant
				_plant->Evolve(*input);
				*newState = _plant->GetState();

				// See if the evolved state is in the specification set and calculate the new norm on the state
				*isInSpecificationSet = _specification.IsInSpecificationSet(*newState);
				if (_useNorm) *norm = (*newState - _specification.GetCenter()).GetWeightedNorm(_normWeights);

				// Find the reinforcing labels and the deterring labels
				Vector reinforcementLabel = Vector(_abstraction->GetInputQuantizer()->GetDimension() * 2);
				Vector deterringLabel = Vector(_abstraction->GetInputQuantizer()->GetDimension() * 2);

				// Create reinforcing and deterring labels
				auto normalInput = _inputQuantizer->NormalizeVector(*input);
				for (unsigned int i = 0; i < _abstraction->GetInputQuantizer()->GetDimension(); i++) {
					reinforcementLabel[i * 2] = normalInput[i];
					reinforcementLabel[i * 2 + 1] = normalInput[i];

					deterringLabel[i * 2] = 0.0;
					deterringLabel[i * 2 + 1] = 1.0;
				}

				// Add labels to the list of labels
				reinforcingLabels->push_back(reinforcementLabel);
				deterringLabels->push_back(deterringLabel);

				break;
			}
		}
	}


	// Run the verification phase
	void Procedure::Verify() {
		_controller.ComputeInputs();

		Log("Verifier", "Computing relevant transitions");
		_verifier->ComputeTransitions();
		Log("Verifier", "Partial abstraction contains " + to_string(_verifier->GetAbstractionCompleteness()) + "% of full abstraction");

		Log("Verifier", "Computing winning set");
		_verifier->ComputeWinningSet();
		Log("Verifier", "Winning set size percentage: " + to_string(_verifier->GetWinningSetPercentage()) + "%");

		if (_computeApparentWinningSet) {
			Log("Verifier", "Computing apparent winning set");
			_verifier->ComputeApparentWinningSet();
			Log("Verifier", "Apparent winning set size percentage: " + to_string(_verifier->GetApparentWinningSetPercentage()) + "%");
		}
		
		// Wait for all the MXNet operations to have finished
		MXNDArrayWaitAll();
	}


	// Adds data to the training queue, if the training queue is full the network will train
	void Procedure::AddToTrainingQueue(vector<Vector> states, vector<Vector> labels) {
		auto episodeSize = states.size();

		// Add states and inputs to the queue
		for (unsigned int i = 0; i < episodeSize; i++) {
			_trainingQueueStates.push_back(states[i]);
			_trainingQueueLabels.push_back(labels[i]);
		}

		// Check if the queue is now overflowing, if so train the neural network using the available data
		auto sizeOfQueue = _trainingQueueStates.size();
		auto neuralNetworkBatchSize = _neuralNetwork->GetBatchSize();

		while (sizeOfQueue >= neuralNetworkBatchSize) {
			vector<Vector> trainingStates;
			vector<Vector> trainingLabels;

			for (unsigned int i = 0; i < neuralNetworkBatchSize; i++) {
				trainingStates.push_back(_trainingQueueStates[0]);
				_trainingQueueStates.erase(_trainingQueueStates.begin());

				trainingLabels.push_back(_trainingQueueLabels[0]);
				_trainingQueueLabels.erase(_trainingQueueLabels.begin());
			}

			_neuralNetwork->Train(trainingStates, trainingLabels);

			sizeOfQueue = _trainingQueueStates.size();
		}
	}


	// Load a neural network
	void Procedure::LoadNeuralNetwork(string path, string name) {
		auto extensionStart = name.find('.');
		auto extension = name.substr(extensionStart + 1, name.size() - extensionStart - 1);

		// Case MATLAB
		if (extension == "m") _fileManager.LoadNetworkFromMATLAB(path, name);
	}


	// Save the neural network
	void Procedure::SaveNetwork(string path) {
		_fileManager.SaveNetworkAsMATLAB(path, "net");
		_fileManager.SaveWinningSetAsMATLAB(path, "dom");
		_fileManager.SaveControllerAsMATLAB(path, "ctl");
		_fileManager.SaveControllerAsStaticController(path, "scs");
		_fileManager.SaveTransitions(path, "trs");
		_fileManager.SaveNetworkAsRaw(path, "raw");
	}


	// Save the neural network as a timestamp
	void Procedure::SaveTimestampedNetwork() {
		// Generate timestamp
		char timestamp[26];
		time_t t = time(0);

		ctime_s(timestamp, sizeof(timestamp), &t);

		string timestampString;
		for (unsigned int i = 0; i < sizeof(timestamp) - 6; i++) {
			if (timestamp[i] != ' ' && timestamp[i] != ':') timestampString += timestamp[i];
		}

		// Concatenate string and chars to form path
		string path = "controllers/timestamps";

		Log("File Manager", "Saving to path: '" + path + "/" + timestampString + "'");

		// Save network to a matlab file under the timestamp
		_fileManager.SaveNetworkAsMATLAB(path, timestampString + "net");
 		_fileManager.SaveWinningSetAsMATLAB(path, timestampString + "dom");
		_fileManager.SaveControllerAsMATLAB(path, timestampString + "ctl");
		_fileManager.SaveControllerAsStaticController(path, timestampString + "scs");
		//_fileManager.SaveAbstractionForSCOTS(path, timestampString + "abss");
		_fileManager.SaveTransitions(path, timestampString + "trs");
		if(_saveRawNeuralNetwork) _fileManager.SaveNetworkAsRaw(path, timestampString + "raw");

		// Log best network
		auto currentWinningDomainPercentage = _verifier->GetWinningSetPercentage();
		if (currentWinningDomainPercentage > _bestControllerWinningDomainPercentage) {
			_bestControllerTimestamp = timestampString;
			_bestControllerWinningDomainPercentage = currentWinningDomainPercentage;
		}

		_fileManager.WriteSynthesisStatusToLog("controllers", "log", _plant->GetName(), timestampString);
	}


	// Get an initial state based on the control specification and episode number
	Vector Procedure::GetInitialStateForTraining(unsigned int episodeCount) {
		float progressionFactor = (float)episodeCount / (float)_maxEpisodes;
		auto initialState = Vector(_abstraction->GetStateQuantizer()->GetDimension());

		// Based on the episode count go through the focuses
		auto amountOfFocuses = _trainingFocuses.size();
		auto currentFocus = _trainingFocuses[episodeCount % amountOfFocuses];

		auto specification = _specification.GetSpecificationType();

		bool validInitialState = false;

		for(size_t attempts = 0; attempts < 10; attempts) {
			switch (currentFocus) {
			case TrainingFocus::SingleState: initialState = _singleStateTrainingFocus; break;
			case TrainingFocus::AllStates:
				if (specification == ControlSpecificationType::Invariance) initialState = _specification.GetVectorFromSpecification();
				else initialState = _stateQuantizer->GetRandomVector();
				break;
			case TrainingFocus::RadialOutwards: initialState = GetVectorRadialFromGoal(_radialInitialStateLower + (_radialInitialStateUpper - _radialInitialStateLower) * progressionFactor); break;
			case TrainingFocus::LosingStates: initialState = _verifier->GetVectorFromLosingDomain(); break;
			case TrainingFocus::NeighboringLosingStates: initialState = _verifier->GetVectorFromLosingNeighborDomain(); break;
			}

			if (specification == ControlSpecificationType::Invariance) {
				if (_specification.IsInSpecificationSet(initialState)) validInitialState = true;
			}
			else if (specification == ControlSpecificationType::Reachability) {
				if (!_specification.IsInSpecificationSet(initialState)) validInitialState = true;
			} else validInitialState = true;

			if (validInitialState) break;
		}

		return initialState;
	}


	// Get a random vector in a radius to the goal based on training time
	Vector Procedure::GetVectorRadialFromGoal(float radius) {
		Vector vector(_stateQuantizer->GetDimension());

		auto goal = _specification.GetCenter();
		auto lowerBound = _stateQuantizer->GetLowerBound();
		auto upperBound = _stateQuantizer->GetUpperBound();

		for (int i = 0; i < _stateQuantizer->GetDimension(); i++) {
			float deltaLower = goal[i] - lowerBound[i];
			float deltaUpper = upperBound[i] - goal[i];

			float spaceSpan = deltaLower + deltaUpper;
			float randomValue = ((float)rand() / RAND_MAX);

			float lowerSpaceProbability = (deltaLower / spaceSpan);

			if (randomValue < lowerSpaceProbability) {
				randomValue = randomValue / lowerSpaceProbability;
				vector[i] = goal[i] - deltaLower * randomValue * radius;
			}
			else {
				randomValue = (randomValue - lowerSpaceProbability) / (1.0 - lowerSpaceProbability);
				vector[i] = goal[i] + deltaUpper * randomValue * radius;
			}
		}

		return _stateQuantizer->QuantizeVector(vector);
	}


	// Log a message 
	void Procedure::Log(string phase, string message) {
		if (phase != _lastLoggedPhase) std::cout << std::endl;
		std::cout << phase << ": \t" << message << std::endl;

		_lastLoggedPhase = phase;
	}
}