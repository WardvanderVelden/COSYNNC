#include "Procedure.h"

namespace COSYNNC {
	// Default constructor
	Procedure::Procedure() { 
		_stateQuantizer = nullptr;
		_inputQuantizer = nullptr;

		_plant = nullptr;

		std::cout << "COSYNNC:\tA correct-by-design neural network controller synthesis procedure." << std::endl;
	}


	// Default deletor
	Procedure::~Procedure() {
		delete _stateQuantizer;
		delete _inputQuantizer;

		delete _plant;

		delete _verifier;
	}


	// Set the state quantizer to the specified quantization parameters
	void Procedure::SpecifyStateQuantizer(Vector eta, Vector lowerBound, Vector upperBound) {
		_stateQuantizer = new Quantizer(true);
		_stateQuantizer->SetQuantizeParameters(eta, lowerBound, upperBound);

		_stateDimension = _stateQuantizer->GetSpaceDimension();
		_stateCardinality = _stateQuantizer->GetCardinality();

		Log("COSYNNC", "State quantizer specified.");
	}


	// Set the input quantizer to the specified quantization parameters
	void Procedure::SpecifyInputQuantizer(Vector eta, Vector lowerBound, Vector upperBound) {
		_inputQuantizer = new Quantizer(true);
		_inputQuantizer->SetQuantizeParameters(eta, lowerBound, upperBound);

		_inputDimension = _inputQuantizer->GetSpaceDimension();
		_inputCardinality = _inputQuantizer->GetCardinality();

		Log("COSYNNC", "Input quantizer specified.");
	}


	// Set the plant
	void Procedure::SetPlant(Plant* plant) {
		_plant = plant;

		Log("COSYNNC", "Plant linked.");
	}


	// Set the neural network
	void Procedure::SetNeuralNetwork(NeuralNetwork* neuralNetwork) {
		_neuralNetwork = neuralNetwork;
		_neuralNetwork->ConfigurateInputOutput(_plant, _inputQuantizer, _maxEpisodeHorizonTrainer, 1.0);

		_outputType = _neuralNetwork->GetOutputType();

		Log("COSYNNC", "Neural network linked.");
	}


	// Set the synthesis parameters
	void Procedure::SpecifySynthesisParameters(unsigned int maxEpisodes, unsigned int maxEpisodeHorizonTrainer, unsigned int verboseEpisode, unsigned int verificationEpisode, unsigned int maxEpisodeHorizonVerifier) {
		_maxEpisodes = maxEpisodes;

		_verboseEpisode = verboseEpisode;
		_verificationEpisode = (verificationEpisode == 0) ? _verboseEpisode * 10 : verificationEpisode;

		_maxEpisodeHorizonTrainer = maxEpisodeHorizonTrainer;
		_maxEpisodeHorizonVerifier = (maxEpisodeHorizonVerifier == 0) ? _maxEpisodeHorizonTrainer * 3 : maxEpisodeHorizonVerifier;

		Log("COSYNNC", "Synthesis parameters specified.");
	}


	// Set the control specification
	void Procedure::SpecifyControlSpecification(ControlSpecificationType type, Vector lowerBound, Vector upperBound) {
		_specification = ControlSpecification(type, _plant);
		_specification.SetHyperInterval(lowerBound, upperBound);

		_controller.SetControlSpecification(&_specification);

		Log("COSYNNC", "Control specification defined.");
	}


	// Specify the radial initial state selection based on the synthesis progression
	void Procedure::SpecifyRadialInitialState(float lower, float upper) {
		_radialInitialStateAvailable = true;

		_radialInitialStateLower = lower;
		_radialInitialStateUpper = (upper == 0.0) ? (1.0 - lower) : upper;

		Log("COSYNNC", "Radial from goal initial state selection specified.");
	}


	// Specify the norm to be used during training
	void Procedure::SpecifyNorm(vector<float> normWeights) {
		_useNorm = true;

		_normWeights = normWeights;

		Log("COSYNNC", "Norm based reinforcement specified.");
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


	// Initialize the procedure, returns false if not all required parameters are specified
	bool Procedure::Initialize() {
		_controller = Controller(_plant, _stateQuantizer, _inputQuantizer);
		_controller.SetNeuralNetwork(_neuralNetwork);
		_controller.SetControlSpecification(&_specification);

		_verifier = new Verifier(_plant, &_controller, _stateQuantizer, _inputQuantizer);
		_verifier->SetVerboseMode(_verboseVerifier);

		_fileManager = FileManager(_neuralNetwork, _verifier, _stateQuantizer, _inputQuantizer, &_specification);

		// TODO: Test if all required parameters are specified before the 

		return true;
	}


	// Run the synthesis procedure
	void Procedure::Synthesize() {
		auto succesfullyInitialized = Initialize();

		if (succesfullyInitialized) {
			Log("COSYNNC", "Procedure was succesfully initialized, synthesize procedure started!");

			Log("Trainer", "Starting training");
			for (int i = 0; i <= _maxEpisodes; i++) {
				Train(i);

				if (i % _verificationEpisode == 0 && i != 0) {
					Log("Verifier", "Performing fixed-point verification.");
					Verify(); 

					Log("File Manager", "Saving neural network with a timestamp.");
					SaveTimestampedNetwork();

					// TODO: Stop if verification is in specs
					Log("Trainer", "Continuing training.");
				}
			}

			// Save the final neural network
			Log("File Manager", "Saving neural network.");
			SaveNetwork("controllers/controller");
		}
		else {
			Log("COSYNNC", "Procedure was not succesfully initialized, check if all required parameters are specified.");
		}

		MXNotifyShutdown();
	}


	// Run the training phase
	void Procedure::Train(unsigned int episodeNumber) {
		if (episodeNumber % _verboseEpisode == 0 && _verboseTrainer) std::cout << std::endl;

		vector<Vector> states;
		vector<Vector> reinforcingLabels;
		vector<Vector> deterringLabels;

		// Get an initial state based on the control specification we are trying to solve for
		auto initialState = GetInitialStateForTraining(episodeNumber);
		_plant->SetState(initialState);

		// Define the norm for determining the networks performance
		float initialNorm = 0.0;
		if(_useNorm) initialNorm = (initialState - _specification.GetCenter()).GetWeightedNorm(_normWeights);
		float norm = initialNorm;

		bool isInSpecificationSet = false;

		// Simulate the episode using the neural network
		for (int j = 0; j < _maxEpisodeHorizonTrainer; j++) {
			auto state = _plant->GetState();

			auto normalizedQuantizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->QuantizeVector(state));
			states.push_back(normalizedQuantizedState);

			// Define vectors to allow access in and outside the switch
			Vector input(_inputDimension);
			Vector networkOutput(_inputDimension);
			Vector newState(_stateDimension);

			// Get data for a single training step
			GetDataForTrainingStep(state, &reinforcingLabels, &deterringLabels, &input, &networkOutput, &newState, &isInSpecificationSet, &norm);

			// DEBUG: Print simulation for verification purposes
			if (episodeNumber % _verboseEpisode == 0 && _verboseTrainer) {
				auto verboseLabels = (_neuralNetwork->GetLabelDimension() <= 5) ? _neuralNetwork->GetLabelDimension() : 5;

				std::cout << "\ti: " << episodeNumber << "\tj: " << j << "\t\tx0: " << newState[0] << "\tx1: " << newState[1];
				for (int i = 0; i < verboseLabels; i++) {
					if (i == 0) std::cout << "\t";
					std::cout << "\tn" << i << ": " << networkOutput[i];;
				}
				std::cout << "\tu: " << input[0] << "\ts: " << isInSpecificationSet << std::endl;
			}

			// Check episode stopping conditions
			bool stopEpisode = false;
			switch (_specification.GetSpecificationType()) {
				case ControlSpecificationType::Invariance: if (!isInSpecificationSet) stopEpisode = true; break;
				case ControlSpecificationType::Reachability: if (isInSpecificationSet) stopEpisode = true; break;
			}

			if (!_stateQuantizer->IsInBounds(newState) || stopEpisode) break;
		}

		// Train the neural network based on the performance of the network
		if ((norm < initialNorm || isInSpecificationSet) && _useNorm) _neuralNetwork->Train(states, reinforcingLabels);
		else if(isInSpecificationSet && !_useNorm) _neuralNetwork->Train(states, reinforcingLabels);
		else _neuralNetwork->Train(states, deterringLabels);
	}


	// Retrieve the training data for a single trianing step
	void Procedure::GetDataForTrainingStep(Vector state, vector<Vector>* reinforcingLabels, vector<Vector>* deterringLabels, Vector* input, Vector* networkOutput, Vector* newState, bool* isInSpecificationSet, float* norm) {
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
				Vector reinforcementLabel = Vector((int)_inputCardinality);
				Vector deterringLabel = Vector((int)_inputCardinality);

				reinforcementLabel = oneHot;

				float sum = 0.0;
				for (int i = 0; i < (int)_inputCardinality; i++) {
					deterringLabel[i] = 1.0 - reinforcementLabel[i];
					sum += deterringLabel[i];
				}

				for (int i = 0; i < (int)_inputCardinality; i++) deterringLabel[i] = deterringLabel[i] / sum;

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
				Vector reinforcementLabel = Vector(_inputDimension * 2);
				Vector deterringLabel = Vector(_inputDimension * 2);

				// Create reinforcing and deterring labels
				auto normalInput = _inputQuantizer->NormalizeVector(*input);
				for (unsigned int i = 0; i < _inputDimension; i++) {
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
		_verifier->ComputeTransitionFunction();
		_verifier->ComputeWinningSet();

		// Empirical random walks
		if (_verboseVerifier) {
			Log("Verifier", "Empirical verification walks");
			for (unsigned int j = 0; j < 5; j++) {
				std::cout << std::endl;
				auto initialState = _stateQuantizer->GetRandomVector();
				_verifier->PrintVerboseWalk(initialState);
			}
			std::cout << std::endl;
		}

		// Log winning set
		Log("Verifier", "Winning set size percentage: " + to_string(_verifier->GetWinningDomainPercentage()) + "%");

		// Wait for all the MXNet operations to have finished
		MXNDArrayWaitAll();
	}


	// Save the neural network
	void Procedure::SaveNetwork(string path) {
		_fileManager.SaveNetworkAsMATLAB(path, "net");
		_fileManager.SaveVerifiedDomainAsMATLAB(path, "dom");
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

		Log("File Manager", "Saving to path: '" + path + "'");

		// Save network to a matlab file under the timestamp
		_fileManager.SaveNetworkAsMATLAB(path, timestampString + "net");
		_fileManager.SaveVerifiedDomainAsMATLAB(path, timestampString + "dom");
	}


	// Log a message 
	void Procedure::Log(string phase, string message) {
		if (phase != _lastLoggedPhase) std::cout << std::endl;
		std::cout << phase << ": \t" << message << std::endl;

		_lastLoggedPhase = phase;
	}


	// Get an initial state based on the control specification and episode number
	Vector Procedure::GetInitialStateForTraining(unsigned int episodeCount) {
		float progressionFactor = (float)episodeCount / (float)_maxEpisodes;
		auto initialState = Vector({ 0.0, 0.0 });

		switch (_specification.GetSpecificationType()) {
		case ControlSpecificationType::Invariance:
			initialState = _specification.GetVectorFromSpecification();
			break;
		case ControlSpecificationType::Reachability:
			initialState = _specification.GetCenter();

			for(unsigned int i = 0; i < 10; i++) {
				//if (episodeCount < _verificationEpisode || episodeCount % 2 == 0) initialState = GetVectorRadialFromGoal(_radialInitialStateLower + progressionFactor * _radialInitialStateUpper);
				//else initialState = _verifier->GetVectorFromLosingDomain();
				//else initialState = _verifier->GetVectorFromLosingNeighborDomain();

				//initialState = _verifier->GetVectorFromLosingNeighborDomain();
				//initialState = GetVectorRadialFromGoal(_radialInitialStateLower + progressionFactor * _radialInitialStateUpper);

				//if (episodeCount % 3 == 0) initialState = GetVectorRadialFromGoal(_radialInitialStateLower + progressionFactor * _radialInitialStateUpper);
				//else if ((episodeCount + 1) % 3 == 0) initialState = _verifier->GetVectorFromLosingDomain();
				//else initialState = _verifier->GetVectorFromLosingNeighborDomain();

				//if (episodeCount % 2 == 0) initialState = _verifier->GetVectorFromLosingNeighborDomain();
				//else initialState = _verifier->GetVectorFromLosingDomain();

				initialState = Vector({ 1.2, 5.3 });

				if (!_specification.IsInSpecificationSet(initialState)) break;
			}
			break;
		}

		return initialState;
	}


	// Get a random vector in a radius to the goal based on training time
	Vector Procedure::GetVectorRadialFromGoal(float radius) {
		Vector vector(_stateQuantizer->GetSpaceDimension());

		auto goal = _specification.GetCenter();
		auto lowerBound = _stateQuantizer->GetSpaceLowerBound();
		auto upperBound = _stateQuantizer->GetSpaceUpperBound();

		for (int i = 0; i < _stateQuantizer->GetSpaceDimension(); i++) {
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

		return vector;
	}
}