#pragma once
#include "mxnet-cpp/MxNetCpp.h"
#include <vector>
#include "Vector.h"
#include "Plant.h"
#include "Quantizer.h"

using namespace mxnet::cpp;

namespace COSYNNC {
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

		// Train the network based on inputs and labels
		virtual void Train(vector<Vector> states, vector<Vector> labels);

		// Print network weights
		void PrintWeights() const;

		// Sets the layers of the network, including the output (if we consider a fully connected network topology)
		void SetLayers(vector<int> layers);

		// Sets the hidden layers of the network
		void SetHiddenLayers(vector<int> hiddenLayers);

	protected:
		vector<int> _layers;
		vector<int> _hiddenLayers;

		int _depth = 0;
		int _batchSize = 1;

		int _inputDimension = 1;
		int _labelDimension = 1;

		bool _verboseOptimizationInspection = false;

		Symbol _network;
		map<string, NDArray> _arguments;
		vector<string> _argumentNames;

		Executor* _executor = NULL;

		Optimizer* _optimizer = NULL;

		// TEMPORARY: For now we are running on the cpu
		Context _context = Context::cpu();
	};
}