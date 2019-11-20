#include "MultilayerPerceptron.h"

using namespace std;

namespace COSYNNC {
	// Initializes a multilayer perceptron neural network topology
	MultilayerPerceptron::MultilayerPerceptron(vector<int> layers, ActivationActType activationFunction) {
		_activationFunction = activationFunction;

		SetLayers(layers);

		InitializeNetworkTopology();
	}

	// Initializes a multilayer perceptron neural network topology
	void MultilayerPerceptron::InitializeNetworkTopology() {
		auto input = Symbol::Variable("input");
		auto label = Symbol::Variable("label");

		_weights = vector<Symbol>(_depth);
		_biases = vector<Symbol>(_depth);
		_outputs = vector<Symbol>(_depth);

		for (int i = 0; i < _depth; ++i) {
			_weights[i] = Symbol::Variable("w" + to_string(i));
			_biases[i] = Symbol::Variable("b" + to_string(i));
			
			Symbol fullyConnected = FullyConnected(
				(i == 0) ? input : _outputs[i - 1],
				_weights[i],
				_biases[i],
				_layers[i]
			);
			
			if (i == (_depth - 1)) {
				_outputs[i] = fullyConnected;
			}
			else {
				_outputs[i] = Activation(fullyConnected, ActivationActType::kRelu);
			}
		}

		// DEBUG: This is now a softmax function (so cross entropy) just to test all the neural network functionality
		_network = SoftmaxOutput(_outputs.back(), label);
	}

	// DEBUG: Temporay test bed for learning MXNET
	void MultilayerPerceptron::Test() {
		const int batchSize = 5;
		const float learningRate = 0.1;
		const float weightDecay = 0.1;
		const int maxEpoch = 10;

		// Generate data
		

		// Arguments for the neural network
		map<string, NDArray> args; 
		args["input"] = NDArray(Shape(batchSize), _context);
		args["label"] = NDArray(Shape(batchSize), _context);

		// Infers the matrix sizes for the network from the netwokr arguments
		_network.InferArgsMap(_context, &args, args);

		// Initialize all parameters with a uniform distribution
		auto initializer = Uniform(-0.1, 0.1);
		for (auto& arg : args) {
			initializer(arg.first, &arg.second);
		}

		// Create an optimizer (for now we will use simply stochastic gradient descent (sgd))
		Optimizer* optimizer = OptimizerRegistry::Find("sgd");
		optimizer->SetParam("rescale_grad", 1.0 / batchSize);
		optimizer->SetParam("lr", learningRate);
		optimizer->SetParam("wd", weightDecay);

		// Bind parameters to the neural network model through an executor
		auto* executor = _network.SimpleBind(_context, args);
		auto argumentNames = _network.ListArguments();

		// Run epochs
		for (int i = 0; i < maxEpoch; ++i) {
			// Reset data

			// While still have data
				// Copy data to network arguments

				// Forward pass
				// Backward pass
				
				// Update parameters that are not input or label
				
		}


		// Make sure to delete the pointers
		delete executor;
		delete optimizer;
	}
}