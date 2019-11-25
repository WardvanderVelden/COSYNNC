#include "MultilayerPerceptron.h"

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
		//_network = SoftmaxOutput(_outputs.back(), label);
		_network = LinearRegressionOutput(_outputs.back(), label);
	}

	// DEBUG: Temporay test bed for learning MXNET
	void MultilayerPerceptron::Test(TrainingData* data, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		const int batchSize = 100;
		const float learningRate = 0.1;
		const float weightDecay = 0.1;
		const int maxEpoch = 10000;
		const int dataSize = data->labels.Size();
		const int steps = 50;

		// Arguments for the neural network (input layer is 2 neurons and the output layer is 1 neuron)
		map<string, NDArray> arguments; 
		arguments["input"] = NDArray(Shape(batchSize, 2), _context);
		arguments["output"] = NDArray(Shape(batchSize, 1), _context);

		// Infers the matrix sizes for the network from the netwokr arguments
		_network.InferArgsMap(_context, &arguments, arguments);

		// Initialize all parameters with a uniform distribution
		auto initializer = Uniform(0.01);
		for (auto& argument : arguments) {
			initializer(argument.first, &argument.second);
		}

		// Create an optimizer (for now we will use simply stochastic gradient descent (sgd))
		Optimizer* optimizer = OptimizerRegistry::Find("adam");
		optimizer->SetParam("rescale_grad", 1.0 / batchSize);
		optimizer->SetParam("lr", learningRate);
		optimizer->SetParam("wd", weightDecay);

		// Bind parameters to the neural network model through an executor
		auto* executor = _network.SimpleBind(_context, arguments);
		auto argumentNames = _network.ListArguments();

		// Train through running epochs
		int currentDataIndex = 0;
		for (int epoch = 0; epoch < maxEpoch; ++epoch) {
			// Reset data
			currentDataIndex = 0;
			std::cout << "Epoch: " << epoch << std::endl;

			// While still have data
			while (currentDataIndex <= (dataSize - batchSize)) {			
				// Get data batch
				NDArray inputBatch = data->inputs.Slice(currentDataIndex, currentDataIndex + batchSize);
				NDArray labelBatch = data->labels.Slice(currentDataIndex, currentDataIndex + batchSize);

				auto inputBatchData = inputBatch.GetData();
				auto labelBatchData = labelBatch.GetData();

				// Copy data to the network
				inputBatch.CopyTo(&arguments["input"]);
				labelBatch.CopyTo(&arguments["output"]);

				// Execute the network
				executor->Forward(true);
				executor->Backward();

				// Update parameters
				for (int i = 0; i < argumentNames.size(); ++i) {
					if (argumentNames[i] == "input" || argumentNames[i] == "output") continue;
					optimizer->Update(i, executor->arg_arrays[i], executor->grad_arrays[i]);
				}

				currentDataIndex += batchSize;
			}
		}
		std::cout << std::endl << "Stopped training!" << std::endl << std::endl;
		system("cls");

		// Evaluate network
		//Accuracy accuracy;
		int correct = 0;

		currentDataIndex = 0;
		while (currentDataIndex <= (dataSize - batchSize)) {
			// Get data batch
			NDArray inputBatch = data->inputs.Slice(currentDataIndex, currentDataIndex + batchSize);
			NDArray labelBatch = data->labels.Slice(currentDataIndex, currentDataIndex + batchSize);

			auto inputBatchShape = inputBatch.GetShape();
			auto labelBatchShape = labelBatch.GetShape();

			// Copy data to the network
			inputBatch.CopyTo(&arguments["input"]);
			labelBatch.CopyTo(&arguments["output"]);

			// Execute the network
			executor->Forward(false);

			auto output = executor->outputs[0];
			auto outputShape = output.GetShape();

			// Determine accuracy
			for (int i = 0; i < batchSize; i++) {
				auto inputValueOne = arguments["input"].At(i, 0);
				auto inputValueTwo = arguments["input"].At(i, 1);
				auto labelValue = labelBatch.At(i, 0);
				//auto predictionValue = arguments["output"].At(i, 0);
				auto predictionValue = output.At(i, 0);

				auto quantizedState = stateQuantizer->QuantizeVector(stateQuantizer->DenormalizeVector(Vector({ inputValueOne, inputValueTwo })));
				auto quantizedPrediction = inputQuantizer->QuantizeVector(inputQuantizer->DenormalizeVector(Vector(predictionValue)));
				auto quantizedLabel = inputQuantizer->QuantizeVector(inputQuantizer->DenormalizeVector(Vector(labelValue)));

				if ((currentDataIndex + i) % steps == 0) std::cout << std::endl;

				std::cout << "i: " << i << "\tx1: " << inputValueOne << "\tx2: " << inputValueTwo << "\tlabel: " << labelValue << "\tpred: " << predictionValue << std::endl;
				//std::cout << "i: " << i << "\tx1: " << quantizedState[0] << "\tx2: " << quantizedState[1] << "\tlabel: " << quantizedLabel[0] << "\tpred: " << quantizedPrediction[0] << std::endl;
				
				if (quantizedLabel == quantizedPrediction) correct++;
			}

			// Update accuracy
			//accuracy.Update(labelBatch, executor->outputs[0]);

			currentDataIndex += batchSize;
		}
		//std::cout << std::endl << "Accuracy: " << accuracy.Get() << std::endl;

		// Accuracy
		float accuracy = (float)correct / (float)dataSize * 100.0;
		std::cout << "Accuracy: " << accuracy << "%" <<  std::endl;

		// Make sure to delete the pointers
		delete executor;
		delete optimizer;
	}

	TrainingData* MultilayerPerceptron::GetTrainingData(Plant* plant, Controller* controller, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		const int episodes = 100;
		const int steps = 10;

		vector<mx_float> inputs;
		vector<mx_float> labels;

		// Simulate episodes
		for (int i = 0; i < episodes; i++) {
			// Set initial position
			controller->ResetController();
			plant->SetState(stateQuantizer->GetRandomVector());

			for (int j = 0; j < steps; j++) {
				Vector quantizedState = stateQuantizer->QuantizeVector(plant->GetState());
				Vector normalizedQuantizedState = stateQuantizer->NormalizeVector(quantizedState);

				Vector controlAction = controller->GetPDControlAction(quantizedState);
				Vector quantizedControlAction = inputQuantizer->QuantizeVector(controlAction);

				Vector normalizedQuantizedControlAction = inputQuantizer->NormalizeVector(quantizedControlAction);

				// Add data 
				inputs.push_back(normalizedQuantizedState[0]);
				inputs.push_back(normalizedQuantizedState[1]);

				labels.push_back(normalizedQuantizedControlAction[0]);

				// Evolve plant with control action
				plant->Evolve(quantizedControlAction);
			}
		}

		TrainingData* data = new TrainingData();
		data->inputs = NDArray(inputs, Shape(episodes*steps, 2), _context);
		data->labels = NDArray(labels, Shape(episodes*steps, 1), _context);
		return data;
	}
}