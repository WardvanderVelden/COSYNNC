#pragma once
#include "Quantizer.h"
#include "Plant.h"
#include "Controller.h"
#include "Verifier.h"

namespace COSYNNC {
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

		// Log a message 
		void Log(string phase = "", string message = "");

		// Get an initial state based on the control specification and episode number
		Vector GetInitialStateForTraining(unsigned int episodeCount);

		// Get a random vector in a radius to the goal based on training time
		Vector GetVectorRadialFromGoal(float radius);
	private:
		Quantizer* _stateQuantizer;
		unsigned int _stateDimension;
		unsigned long _stateCardinality;

		Quantizer* _inputQuantizer;
		unsigned int _inputDimension;
		unsigned long _inputCardinality;

		Plant* _plant;

		Verifier* _verifier;

		NeuralNetwork* _neuralNetwork;
		OutputType _outputType;

		Controller _controller;

		ControlSpecification _specification;

		// Synthesis parameters
		unsigned int _maxEpisodeHorizonTrainer;
		unsigned int _maxEpisodeHorizonVerifier;

		unsigned int _maxEpisodes;

		unsigned int _verboseEpisode;
		unsigned int _verificationEpisode;

		bool _radialInitialStateAvailable = false;
		float _radialInitialStateLower = 0.0;
		float _radialInitialStateUpper = 1.0;

		bool _useNorm = false;
		vector<float> _normWeights = { 1.0, 1.0 };

		// Debug and logging parameters
		bool _verboseMode = true;
	};
}
