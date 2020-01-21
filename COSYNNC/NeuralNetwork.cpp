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
	void NeuralNetwork::InitializeOptimizer(string optimizer, float learningRate, float weightDecayRate, bool verboseOptimizationInspection) {
		_verboseOptimizationInspection = verboseOptimizationInspection;

		_optimizer = OptimizerRegistry::Find(optimizer);
		_optimizer->SetParam("rescale_grad", 1.0 / _batchSize);
		_optimizer->SetParam("lr", learningRate);
		_optimizer->SetParam("wd", weightDecayRate);
	}
	

	// Configures the neural network to receive input and output data compatible with the state and input dimensions and batch size
	void NeuralNetwork::ConfigurateInputOutput(Plant* plant, Quantizer* inputQuantizer, int batchSize, float initialDistribution) {
		_inputDimension = plant->GetStateSpaceDimension();

		// Define label dimension based on output type
		switch (_outputType) {
		case OutputType::Labelled:
			_labelDimension = inputQuantizer->GetCardinality(); // Neuron for every potential input
			break;
		case OutputType::Range:
			_labelDimension = inputQuantizer->GetSpaceDimension() * 2; // Two neurons per axis
			break;
		}

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

		NDArray networkInput(data, Shape(_batchSize, _inputDimension), _context);
		networkInput.WaitToRead();

		networkInput.CopyTo(&_arguments["input"]);
		networkInput.WaitToWrite();

		MXNDArrayWaitAll();

		_executor->Forward(false);

		auto outputDimension = _layers.back();
		Vector output(outputDimension);
		for (int i = 0; i < outputDimension; i++) {
			output[i] = _executor->outputs[0].At(0, i);
		}

		return output;
	}


	// Evaluates the neural network in batch
	Vector* NeuralNetwork::EvaluateNetworkInBatch(Vector* inputs, unsigned int batchSize) {
		// Format data
		vector<mx_float> data;

		for (unsigned int i = 0; i < _batchSize; i++) {
				auto input = inputs[i % batchSize];
				for (unsigned int j = 0; j < _inputDimension; j++)
					data.push_back(input[j]);
		}

		NDArray networkInput(data, Shape(_batchSize, _inputDimension), _context);
		networkInput.WaitToRead();

		networkInput.CopyTo(&_arguments["input"]);
		networkInput.WaitToWrite();

		MXNDArrayWaitAll();

		// Execute network operations
		_executor->Forward(false);

		// Get data
		Vector* outputs = new Vector[batchSize];
		auto outputDimension = _layers.back();
		for (unsigned int i = 0; i < batchSize; i++) {
			outputs[i] = Vector(outputDimension);
			for (unsigned int j = 0; j < outputDimension; j++) {
				outputs[i][j] = _executor->outputs[0].At(i, j);
			}
		}

		return outputs;
	}


	// Train the network based on inputs and labels
	void NeuralNetwork::Train(vector<Vector> states, vector<Vector> labels) {
		if (_optimizer == NULL) return;

		// Format data to suit the MX library
		vector<mx_float> inputData;
		vector<mx_float> labelData;

		// TODO: Make it so that it does not repeat during training, as this emphesizes short episodes 
		auto amountOfStates = states.size();
		for (int i = 0; i < _batchSize; i++) {
			auto state = states[i % amountOfStates];
			for (int j = 0; j < _inputDimension; j++)
				inputData.push_back(state[j]);

			auto label = labels[i % amountOfStates];
			for (int j = 0; j < _labelDimension; j++)
				labelData.push_back(label[j]);
		}

		NDArray networkInputData(inputData, Shape(_batchSize, _inputDimension), _context);
		NDArray networkLabelData(labelData, Shape(_batchSize, _labelDimension), _context);

		networkInputData.WaitToRead();
		networkLabelData.WaitToRead();

		// Assign the data to the network
		networkInputData.CopyTo(&_arguments["input"]);
		networkLabelData.CopyTo(&_arguments["label"]);

		networkInputData.WaitToWrite(); 
		networkLabelData.WaitToWrite();

		MXNDArrayWaitAll();

		// Train
		_executor->Forward(true);
		_executor->Backward();

		// DEBUG: Print to debug
		if (_verboseOptimizationInspection) {
			for (int i = 0; i < _batchSize; i++) {
				std::cout << "\tx0: " << _arguments["input"].At(i, 0) << "\tx1: " << _arguments["input"].At(i, 1) << "\tp: " << _executor->outputs[0].At(i, 0) << "\tl: " << _arguments["label"].At(i, 0) << std::endl;
			}
		}

		// Update parameters
		for (int i = 0; i < _argumentNames.size(); ++i) {
			if (_argumentNames[i] == "input" || _argumentNames[i] == "label") continue;
			_optimizer->Update(i, _executor->arg_arrays[i], _executor->grad_arrays[i]);
		}

		_justTrained = true;
	}


	// Saves the current network
	void NeuralNetwork::Save(string path) {
		// If the path is null create a timestamp
		if (path == "") {
			char timestamp[26];
			time_t t = time(0);

			ctime_s(timestamp, sizeof(timestamp), &t);

			string timestampString;
			for (unsigned int i = 0; i < sizeof(timestamp) - 2; i++) {
				if (timestamp[i] != ' ') timestampString += timestamp[i];
			}

			std::cout << "Generated neural network name: " << path << std::endl;

			// Concatenate string and chars to form path
			path = "networks/";
			path += timestampString;
			path += ".nn";
		}

		_network.Save(path);
	}


	// Loads a network
	void NeuralNetwork::Load(string path) {
		if (path == "") std::cout << "Attempted to load a network without specifying path!" << std::endl;

		_network = mxnet::cpp::Symbol::Load(path);
	}


	// Print network weights
	void NeuralNetwork::PrintWeights() const {
		for (int i = 0; i < _argumentNames.size(); i++) {
			auto name = _argumentNames[i];
			std::cout << name << std::endl;
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


	// Sets the output type of the network
	void NeuralNetwork::SetOutputType(OutputType outputType) {
		_outputType = outputType;
	}


	// Returns the output type of the network
	OutputType NeuralNetwork::GetOutputType() const {
		return _outputType;
	}


	// Returns the batch size of the network
	int NeuralNetwork::GetBatchSize() const {
		return _batchSize;
	}


	// Returns the label dimension
	int NeuralNetwork::GetLabelDimension() const {
		return _labelDimension;
	}
}