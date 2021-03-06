#include "MultilayerPerceptron.h"

namespace COSYNNC {
	// Initializes a neural network of the multilayer perceptron topology
	MultilayerPerceptron::MultilayerPerceptron(vector<int> hiddenLayers, ActivationActType activationFunction, LossFunctionType lossFunction) {
		_activationFunction = activationFunction;
		_lossFunction = lossFunction;

		SetHiddenLayers(hiddenLayers);
	}


	// Initializes a neural network of the multilayer perceptron topology
	MultilayerPerceptron::MultilayerPerceptron(vector<int> hiddenLayers, ActivationActType activationFunction, OutputType outputType) {
		_activationFunction = activationFunction;
		_outputType = outputType;

		switch (_outputType) {
			case OutputType::Labelled: _lossFunction = LossFunctionType::CrossEntropy; break;
			case OutputType::Range: _lossFunction = LossFunctionType::Proportional; break;
		}

		SetHiddenLayers(hiddenLayers);
	}

	// Initializes a multilayer perceptron neural network topology
	MultilayerPerceptron::MultilayerPerceptron(vector<int> hiddenLayers, ActivationActType activationFunction, LossFunctionType lossFunction, OutputType outputType) {
		_activationFunction = activationFunction;
		_lossFunction = lossFunction;
		_outputType = outputType;

		SetHiddenLayers(hiddenLayers);
	}

	// Initializes a multilayer perceptron neural network topology
	void MultilayerPerceptron::InitializeGraph() {
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
			
			// Define activation function based on the loss function
			switch (_lossFunction) {
			case LossFunctionType::CrossEntropy:
				if (i == (_depth - 1)) {
					_outputs[i] = fullyConnected;
				}
				else {
					_outputs[i] = Activation(fullyConnected, ActivationActType::kRelu);
				}
				//_outputs[i] = Activation(fullyConnected, _activationFunction);
				break;
			case LossFunctionType::Proportional:
				_outputs[i] = Activation(fullyConnected, _activationFunction);
				break;
			case LossFunctionType::Log:
				_outputs[i] = Activation(fullyConnected, ActivationActType::kSigmoid);
				break;
			}
		}

		// Define cost function
		switch (_lossFunction) {
		case LossFunctionType::CrossEntropy:
			_network = SoftmaxOutput(_outputs.back(), label);
			break;
		case LossFunctionType::Proportional:
		case LossFunctionType::Quadratic:
			_network = LinearRegressionOutput(_outputs.back(), label);
			break;
		case LossFunctionType::Log:
			_network = LogisticRegressionOutput(_outputs.back(), label);
			break;
		}
	}
}