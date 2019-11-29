#include "NeuralNetwork.h"

namespace COSYNNC {
	// Initializes a default neural network
	NeuralNetwork::NeuralNetwork() {

		
	}


	// Destructor for the neural network
	NeuralNetwork::~NeuralNetwork() {
		delete _executor;
		delete _optimizer;
	}


	// Initialize the neural network topology
	void NeuralNetwork::InitializeGraph() {	}

	
	// Initializes the optimizer for training
	void NeuralNetwork::InitializeOptimizer(string optimizer, float learningRate, float weightDecayRate) {
		_optimizer = OptimizerRegistry::Find(optimizer);
		_optimizer->SetParam("rescale_grad", 1.0 / _batchSize);
		_optimizer->SetParam("lr", learningRate);
		_optimizer->SetParam("wd", weightDecayRate);
	}
	

	// Configures the neural network to receive input and output data compatible with the state and input dimensions and batch size
	void NeuralNetwork::ConfigurateInputOutput(Plant* plant, int batchSize, float initialDistribution) {
		_inputDimension = plant->GetStateSpaceDimension();
		_labelDimension = plant->GetInputSpaceDimension();

		// Define layers based on label dimension
		auto layers = vector<int>(_hiddenLayers);
		layers.push_back(_labelDimension);
		SetLayers(layers);

		// Defines the dimensions of the input and output of the neural network based on the plant and the batch size
		_batchSize = batchSize;

		_arguments["input"] = NDArray(Shape(batchSize, _inputDimension), _context);
		_arguments["label"] = NDArray(Shape(batchSize, _labelDimension), _context);

		// Initialize the neural network graph
		InitializeGraph();

		// Infers the size of the other matrices and vectors based on the input and output and amount of neurons per layer
		_network.InferArgsMap(_context, &_arguments, _arguments);

		// Initialize all parameters with a uniform distribution
		auto initializer = Uniform(initialDistribution);
		for (auto& argument : _arguments) {
			initializer(argument.first, &argument.second);
		}

		// Bind parameters to the neural network model through an executor
		_executor = _network.SimpleBind(_context, _arguments);
		_argumentNames = _network.ListArguments();
	}


	// Evaluates the neural network on a single input
	Vector NeuralNetwork::EvaluateNetwork(Vector input) {
		vector<mx_float> data;
		for (int i = 0; i < input.GetLength(); i++) data.push_back(input[i]);

		NDArray networkInput(data, Shape(_batchSize, 2), _context);
		networkInput.CopyTo(&_arguments["input"]);
		networkInput.WaitToWrite(); // DEBUG: May also need a wait to read to prevent memory leaks

		_executor->Forward(false);

		auto outputDimension = _layers.back();
		Vector output(outputDimension);
		for (int i = 0; i < outputDimension; i++)
			output[i] = _executor->outputs[0].At(0, i);

		// DEBUG: Print network inputs and outputs to confirm behaviour
		//std::cout << "\tx0: " << _arguments["input"].At(0, 0) << "\tx1: " << _arguments["input"].At(0, 1) << "\tp: " << _executor->outputs[0].At(0,0) << std::endl;

		return output;
	}


	// Train the network based on inputs and labels
	void NeuralNetwork::Train(vector<Vector> states, vector<Vector> labels) {
		if (_optimizer == NULL) return;

		// Format data to suit the MX library
		vector<mx_float> inputData;
		vector<mx_float> labelData;

		for (int i = 0; i < states.size(); i++) {
			auto state = states[i];
			for (int j = 0; j < _inputDimension; j++)
				inputData.push_back(state[j]);

			auto label = labels[i];
			for (int j = 0; j < _labelDimension; j++)
				labelData.push_back(label[j]);
		}

		NDArray networkInputData(inputData, Shape(_batchSize, _inputDimension), _context);
		NDArray networkLabelData(labelData, Shape(_batchSize, _labelDimension), _context);

		// Assign the data to the network
		networkInputData.CopyTo(&_arguments["input"]);
		networkLabelData.CopyTo(&_arguments["label"]);

		// DEBUG: May also need a wait to read to prevent memory leaks
		networkInputData.WaitToWrite(); 
		networkLabelData.WaitToWrite();

		/*for (int i = 0; i < _batchSize; i++) {
			std::cout << "\tx0: " << _arguments["input"].At(i, 0) << "\tx1: " << _arguments["input"].At(i, 1) << std::endl;
		}*/

		// Train
		_executor->Forward(true);
		_executor->Backward();

		// Update parameters
		for (int i = 0; i < _argumentNames.size(); ++i) {
			if (_argumentNames[i] == "input" || _argumentNames[i] == "label") continue;
			_optimizer->Update(i, _executor->arg_arrays[i], _executor->grad_arrays[i]);
		}
	}


	// Sets the layers of the network, including the output (if we consider a fully connected network topology)
	void NeuralNetwork::SetLayers(vector<int> layers) {
		_layers = layers;
		_depth = layers.size();
	}


	// Sets the hidden layers of the network
	void NeuralNetwork::SetHiddenLayers(vector<int> hiddenLayers) {
		_hiddenLayers = hiddenLayers;
		_depth = hiddenLayers.size() + 1;
	}
}