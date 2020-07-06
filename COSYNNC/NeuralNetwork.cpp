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
	void NeuralNetwork::ConfigureInputOutput(Plant* plant, Quantizer* inputQuantizer, int batchSize, float initialDistribution) {
		_inputDimension = plant->GetStateSpaceDimension();

		// Define label dimension based on output type
		switch (_outputType) {
		case OutputType::Labelled:
			_labelDimension = inputQuantizer->GetCardinality(); // Neuron for every potential input
			break;
		case OutputType::Range:
			_labelDimension = inputQuantizer->GetDimension() * 2; // Two neurons per axis
			break;
		}

		Initialize(batchSize, initialDistribution);
	}


	// Configures a generic neural network
	void NeuralNetwork::ConfigureInputOutput(unsigned int inputNeurons, unsigned int outputNeurons, unsigned int batchSize, float initialDistribution) {
		_inputDimension = inputNeurons;
		_labelDimension = outputNeurons;

		Initialize(batchSize, initialDistribution);
	}


	// Initializes the topology and graph based on the hidden layers and input and label dimension
	void NeuralNetwork::Initialize(unsigned int batchSize, float initialDistribution) {
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
			output[i] = min(max(_executor->outputs[0].At(0, i), (mx_float)0.0), (mx_float)1.0);
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
				//outputs[i][j] = _executor->outputs[0].At(i, j);
				outputs[i][j] = min(max(_executor->outputs[0].At(i, j), (mx_float)0.0), (mx_float)1.0);
			}
		}

		return outputs;
	}


	// Evaluates the neural network in batch
	Vector* NeuralNetwork::EvaluateNetworkInBatch(vector<Vector> inputs) {
		const unsigned int batchSize = inputs.size();

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
				//outputs[i][j] = min(max(_executor->outputs[0].At(i, j), (mx_float)0.0), (mx_float)1.0);
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

		auto amountOfStates = states.size();
		for (int i = 0; i < _batchSize; i++) {
			auto state = states[i % amountOfStates];
			for (int j = 0; j < _inputDimension; j++) inputData.push_back(state[j]);

			auto label = labels[i % amountOfStates];
			for (int j = 0; j < _labelDimension; j++) labelData.push_back(label[j]);
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

		// Update parameters
		for (int i = 0; i < _argumentNames.size(); ++i) {
			if (_argumentNames[i] == "input" || _argumentNames[i] == "label") continue;

			auto arguments = _executor->arg_arrays[i];
			auto gradients = _executor->grad_arrays[i];

			_optimizer->Update(i, arguments, gradients);
		}

		_justTrained = true;
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


	// Sets the value of an argument in the neural network
	void NeuralNetwork::SetArgument(string name, vector<mx_float> data) {
		NDArray ndarray(data, Shape(GetArgumentShape(name)), _context);
		ndarray.WaitToRead();

		ndarray.CopyTo(&_arguments[name]);
		ndarray.WaitToWrite();
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
	int NeuralNetwork::GetInputDimension() const {
		return _inputDimension;
	}


	// Returns the label dimension
	int NeuralNetwork::GetLabelDimension() const {
		return _labelDimension;
	}


	// Returns the layer depth
	int NeuralNetwork::GetLayerDepth() const {
		return _depth;
	}


	// Returns the layers
	vector<int> NeuralNetwork::GetLayers() const {
		return _layers;
	}


	// Returns the list of argument names
	vector<string> NeuralNetwork::GetArgumentNames() const {
		return _argumentNames;
	}


	// Returns the shape of an argument in the network
	vector<index_t> NeuralNetwork::GetArgumentShape(string name) {
		auto argument = _arguments[name];
		return argument.GetShape();
	}


	// Returns the data size of the neural network in bytes
	int NeuralNetwork::GetDataSize() {
		unsigned int floats = 0;

		for (unsigned int i = 0; i < _arguments.size(); i++) {
			auto argumentName = _argumentNames[i];
			if (argumentName == "input" || argumentName == "label") continue;

			auto argument = _arguments[argumentName];

			auto shape = argument.GetShape();

			unsigned int argumentFloats = 0;
			for (unsigned int j = 0; j < shape.size(); j++) {
				if (j == 0) argumentFloats += shape[j];
				else argumentFloats *= shape[j];
			}

			floats += argumentFloats;
		}

		_dataSize = floats * 4;
		_dataSize += 2 + _depth; // data required to emply the matrix structure

		return _dataSize;
	}


	// Returns a value in the argument
	mx_float NeuralNetwork::GetArgumentValue(string name, vector<unsigned int> index) {
		auto argument = _arguments[name];

		// Vector
		if (index.size() == 1) {
			return argument.At(index[0]);
		}
		// Matrix
		else { 
			return argument.At(index[0], index[1]);
		}
	}
}