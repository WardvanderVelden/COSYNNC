#pragma once
#include "Quantizer.h"
#include "Plant.h"
#include "Controller.h"
#include "Verifier.h"
#include "FileManager.h"

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
	};

	class Procedure {
	public:
		// Default constructor
		Procedure();

		// Default deletor
		~Procedure();

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

		// Set the plant
		void SetPlant(Plant* plant);

		// Set the neural network
		void SetNeuralNetwork(NeuralNetwork* neuralNetwork);

		// Initialize the procedure, returns false if not all required parameters are specified
		bool Initialize();

		// Run the synthesis procedure
		void Synthesize();

		// Run the training phase
		void Train(unsigned int episodeNumber);

		// Retrieve the training data for a single trianing step
		void GetDataForTrainingStep(Vector state, vector<Vector>* reinforcingLabels, vector<Vector>* deterringLabels, Vector* input, Vector* networkOutput, Vector* newState, bool* isInSpecificationSet, float* norm);

		// Run the verification phase
		void Verify();

		// Save the neural network
		void SaveNetwork(string path = "");

		// Save the neural network as a timestamp
		void SaveTimestampedNetwork();
		
		// Log a message 
		void Log(string phase = "", string message = "");

		// Get an initial state based on the control specification and episode number
		Vector GetInitialStateForTraining(unsigned int episodeCount);

		// Get a random vector in a radius to the goal based on training time
		Vector GetVectorRadialFromGoal(float radius);
	private:
		Quantizer* _stateQuantizer;
		unsigned int _stateDimension = 1;
		unsigned long _stateCardinality = 0;

		Quantizer* _inputQuantizer;
		unsigned int _inputDimension = 1;
		unsigned long _inputCardinality = 0;

		Plant* _plant;

		Verifier* _verifier;

		NeuralNetwork* _neuralNetwork;
		OutputType _outputType;

		Controller _controller;

		ControlSpecification _specification;

		FileManager _fileManager;

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

		// Debug and logging parameters
		bool _verboseTrainer = true;
		bool _verboseVerifier = true;
		string _lastLoggedPhase = "";
	};
}
