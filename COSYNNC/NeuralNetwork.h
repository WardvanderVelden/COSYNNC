#pragma once
#include "mxnet-cpp/MxNetCpp.h"
#include <vector>
#include "Vector.h"
#include "Plant.h"
#include "Quantizer.h"

#include <time.h>
#include <stdlib.h>

using namespace mxnet::cpp;

namespace COSYNNC {
	enum class LossFunctionType {
		CrossEntropy,
		Proportional,
		Quadratic
	};

	enum class OutputType {
		Labelled,
		Range
	};

	class NeuralNetwork {
	public:
		// Initializes a default neural network
		NeuralNetwork();

		// Destructor for the neural network
		~NeuralNetwork();

		// Initialize the neural network topology
		virtual void InitializeGraph();

		// Initializes the optimizer for training
		virtual void InitializeOptimizer(string optimizer = "sgd", float learningRate = 0.1, float weightDecayRate = 0.1, bool verboseOptimizationInspection = false);

		// Configures the neural network to receive input and output data compatible with the state and input dimensions and batch size
		virtual void ConfigurateInputOutput(Plant* plant, Quantizer* inputQuantizer, int batchSize = 1, float initialDistribution = 0.1);

		// Evaluates the neural network
		virtual Vector EvaluateNetwork(Vector input);

		// Evaluates the neural network in batch
		virtual Vector* EvaluateNetworkInBatch(Vector* inputs, unsigned int batchSize);

		// Train the network based on inputs and labels
		virtual void Train(vector<Vector> states, vector<Vector> labels);

		// Saves the current network
		virtual void Save(string path = "");

		// Loads a network
		virtual void Load(string path = "");

		// Print network weights
		void PrintWeights() const;

		// Sets the layers of the network, including the output (if we consider a fully connected network topology)
		void SetLayers(vector<int> layers);

		// Sets the hidden layers of the network
		void SetHiddenLayers(vector<int> hiddenLayers);

		// Sets the output type of the network
		void SetOutputType(OutputType outputType);

		// Returns the output type of the network
		OutputType GetOutputType() const;

		// Returns the batch size of the network
		int GetBatchSize() const;

		// Returns the label dimension
		int GetLabelDimension() const;

	protected:
		vector<int> _layers;
		vector<int> _hiddenLayers;

		int _depth = 0;
		int _batchSize = 1;

		int _inputDimension = 1;
		int _labelDimension = 1;

		bool _verboseOptimizationInspection = false;

		bool _justTrained = true;

		Symbol _network;
		map<string, NDArray> _arguments;
		vector<string> _argumentNames;

		Executor* _executor = NULL;

		Optimizer* _optimizer = NULL;

		LossFunctionType _lossFunction = LossFunctionType::Proportional;
		OutputType _outputType = OutputType::Labelled;

		// TEMPORARY: For now we are running on the cpu
		Context _context = Context::cpu();
	};
}