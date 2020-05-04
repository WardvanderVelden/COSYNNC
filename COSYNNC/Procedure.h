#pragma once
#include "Quantizer.h"
#include "Plant.h"
#include "Controller.h"
#include "Verifier.h"
#include "FileManager.h"
#include "BddManager.h"

namespace COSYNNC {
	enum class TrainingFocus {
		SingleState,
		AllStates,
		RadialOutwards,
		LosingStates,
		NeighboringLosingStates,

		AlternatingRadialSingle,
		AlternatingRadialLosing,
		AlternatingRadialNeighboringLosing,
		AlternatingRadialLosingNeighborLosing,
	};

	class Procedure {
	public:
		// Default constructor
		Procedure();

		// Default destructor
		~Procedure();

		#pragma region Procedure Specifiers

		// Specify the state quantizer to the specified quantization parameters
		void SpecifyStateQuantizer(Vector eta, Vector lowerBound, Vector upperBound);

		// Specify the input quantizer to the specified quantization parameters
		void SpecifyInputQuantizer(Vector eta, Vector lowerBound, Vector upperBound);

		// Specify the control specification
		void SpecifyControlSpecification(ControlSpecificationType type, Vector lowerBound, Vector upperBound);

		// Specify the synthesis parameters
		void SpecifySynthesisParameters(unsigned int maxEpisodes = 1000000, unsigned int maxEpisodeHorizonTrainer = 15, unsigned int verboseEpisode = 2500, unsigned int verificationEpisode = 0, unsigned int maxEpisodeHorizonVerifier = 0);

		// Specify the radial initial state selection based on the synthesis progression
		void SpecifyRadialInitialState(float lower, float upper = 0.0);

		// Specify the norm to be used during training
		void SpecifyNorm(vector<float> normWeights);

		// Specify the verbosity of the procedure
		void SpecifyVerbosity(bool verboseTrainer = false, bool verboseVerifier = false);

		// Specify the training focus that should be used during training
		void SpecifyTrainingFocus(TrainingFocus trainingFocus, Vector singleStateTrainingFocus = Vector((unsigned int)0));

		// Specify if the network should reinforce upon reaching the winning set (only applicable to reachability)
		void SpecifyWinningSetReinforcement(bool reinforce = false);

		// Specify if the raw neural network should be saved
		void SpecifySaveRawNeuralNetwork(bool save = false);

		// Specify if the apparent winning set should be computed
		void SpecifyComputeApparentWinningSet(bool compute = false);

		// Specify what type of transition calculation should be used
		void SpecifyUseRefinedTransitions(bool useRefined = true);

		// Set the plant
		void SetPlant(Plant* plant);

		// Set the neural network
		void SetNeuralNetwork(NeuralNetwork* neuralNetwork);

		#pragma endregion Procedure Specifiers

		// Initialize the procedure, returns false if not all required parameters are specified
		bool Initialize();

		// Run the synthesis procedure
		void Synthesize();

		// Run the verification phase
		void Verify();

		// Load a neural network
		void LoadNeuralNetwork(string path, string name);

		// Save the neural network
		void SaveNetwork(string path = "");

		// Save the neural network as a timestamp
		void SaveTimestampedNetwork();
	private:
		// Iterate the training episode
		void IterateEpisode(unsigned int episodeNumber);

		// // Formats the training data for a single training step
		void FormatTrainingData(Vector state, vector<Vector>* reinforcingLabels, vector<Vector>* deterringLabels, Vector* input, Vector* networkOutput, Vector* newState, bool* isInSpecificationSet, float* norm);

		// Adds data to the training queue, if the network is full the network will train
		void AddToTrainingQueue(vector<Vector> states, vector<Vector> labels);

		// Get an initial state based on the control specification and episode number
		Vector GetInitialStateForTraining(unsigned int episodeCount);

		// Get a random vector in a radius to the goal based on training time
		Vector GetVectorRadialFromGoal(float radius);

		// Log a message 
		void Log(string phase = "", string message = "");


		Abstraction* _abstraction = nullptr;

		Plant* _plant = nullptr;
		Controller _controller;

		Quantizer* _stateQuantizer = nullptr;
		Quantizer* _inputQuantizer = nullptr;

		ControlSpecification _specification;

		Verifier* _verifier = nullptr;

		NeuralNetwork* _neuralNetwork = nullptr;
		OutputType _outputType = OutputType::Labelled;

		FileManager _fileManager;
		BddManager _bddManager;

		// Training queue
		vector<Vector> _trainingQueueStates;
		vector<Vector> _trainingQueueLabels;

		// Synthesis parameters
		unsigned int _maxEpisodeHorizonTrainer = 10;
		unsigned int _maxEpisodeHorizonVerifier = 50;

		unsigned int _maxEpisodes = 100000;

		unsigned int _verboseEpisode = 2500;
		unsigned int _verificationEpisode = 25000;

		bool _radialInitialStateAvailable = false;
		float _radialInitialStateLower = 0.0;
		float _radialInitialStateUpper = 1.0;

		TrainingFocus _trainingFocus;
		Vector _singleStateTrainingFocus;

		bool _useNorm = false;
		vector<float> _normWeights = { 1.0, 1.0 };

		bool _useWinningSetReinforcement = false;

		bool _computeApparentWinningSet = false;
		bool _saveRawNeuralNetwork = false;

		bool _useRefinedTransitions = false;

		// Controller log
		float _bestControllerWinningDomainPercentage = 0.0;
		string _bestControllerTimestamp = "";

		// Debug and logging parameters
		bool _verboseTrainer = true;
		bool _verboseVerifier = true;
		string _lastLoggedPhase = "";

		bool _hasSuccesfullyInitialized = false;
	};
}
