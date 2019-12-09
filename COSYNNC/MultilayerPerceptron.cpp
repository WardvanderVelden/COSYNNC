#include "MultilayerPerceptron.h"

namespace COSYNNC {
	// Initializes a multilayer perceptron neural network topology
	MultilayerPerceptron::MultilayerPerceptron(vector<int> hiddenLayers, ActivationActType activationFunction) {
		_activationFunction = activationFunction;

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
			
			/*if (i == (_depth - 1)) {
				_outputs[i] = fullyConnected;
			}
			else {
				_outputs[i] = Activation(fullyConnected, ActivationActType::kRelu);
			}*/
			_outputs[i] = Activation(fullyConnected, _activationFunction);
		}

		_network = LinearRegressionOutput(_outputs.back(), label);
	}

	// DEBUG: Temporay test bed for learning MXNET
	void MultilayerPerceptron::Test(TrainingData* data, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		const int batchSize = 2500;
		const float learningRate = 0.05;
		const float weightDecay = 0.0001;

		const int maxEpoch = 1000;
		const int verbalEpoch = 10;

		const int dataSize = data->labels.Size();

		// Arguments for the neural network (input layer is 2 neurons and the output layer is 1 neuron)
		map<string, NDArray> arguments; 
		arguments["input"] = NDArray(Shape(batchSize, 2), _context);
		arguments["label"] = NDArray(Shape(batchSize, 1), _context);

		// Infers the matrix sizes for the network from the netwokr arguments
		_network.InferArgsMap(_context, &arguments, arguments);

		// Initialize all parameters with a uniform distribution
		auto initializer = Uniform(0.1);
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
		std::cout << "Phase: Training" << std::endl;

		int dataIndex = 0;
		for (int epoch = 0; epoch < maxEpoch; ++epoch) {
			// Reset data
			dataIndex = 0;

			// While still have data
			while (dataIndex <= (dataSize - batchSize)) {			
				// Get data batch
				NDArray inputBatch = data->inputs.Slice(dataIndex, dataIndex + batchSize);
				NDArray labelBatch = data->labels.Slice(dataIndex, dataIndex + batchSize);

				// Synchronization step to prevent memory leak
				inputBatch.WaitToRead();
				labelBatch.WaitToRead();

				// Copy data to the network
				inputBatch.CopyTo(&arguments["input"]);
				labelBatch.CopyTo(&arguments["label"]);

				// Synchronization step to prevent memory leak
				inputBatch.WaitToWrite();
				labelBatch.WaitToWrite();

				// Execute the network
				executor->Forward(true);
				executor->Backward();

				// Update parameters
				for (int i = 0; i < argumentNames.size(); ++i) {
					auto shape = executor->arg_arrays[i].GetShape();
					auto name = argumentNames[i];

					if (argumentNames[i] == "input" || argumentNames[i] == "label") continue;
					optimizer->Update(i, executor->arg_arrays[i], executor->grad_arrays[i]);	
				}

				// Print labeled test data and current prediction
				if (dataIndex == 0 && epoch % verbalEpoch == 0) {
					std::cout << "\tEpoch: " << epoch << std::endl;

					int randomIndex = rand() % batchSize;

					auto input = Vector({ arguments["input"].At(randomIndex, 0) , arguments["input"].At(randomIndex, 1) });
					//auto input = arguments["input"].At(randomIndex, 0);
					auto label = arguments["label"].At(randomIndex, 0);
					auto prediction = executor->outputs[0].At(randomIndex, 0);
					auto difference = abs(label - prediction);

					std::cout << "\t\tx0: " << input[0] << "\tx1: " << input[1] << "\tl: " << label << "\tp: " << prediction << "\td: " << difference << std::endl;
					//std::cout << "\t\tx0: " << input << "\tl: " << label << "\tp: " << prediction << std::endl;
				}

				dataIndex += batchSize;
			}
		}
		std::cout << "\tStopped training" << std::endl << std::endl;


		// Evaluate network
		Accuracy accuracy;
		int correctDataPoints = 0;

		std::cout << "Phase: Validating" << std::endl;

		dataIndex = 0;
		while (dataIndex <= (dataSize - batchSize)) {
			// Get data batch
			NDArray inputBatch = data->inputs.Slice(dataIndex, dataIndex + batchSize);
			NDArray labelBatch = data->labels.Slice(dataIndex, dataIndex + batchSize);

			auto inputBatchShape = inputBatch.GetShape();
			auto labelBatchShape = labelBatch.GetShape();

			// Copy data to the network
			inputBatch.CopyTo(&arguments["input"]);
			labelBatch.CopyTo(&arguments["label"]);

			// Execute the network
			executor->Forward(false);

			// Determine accuracy
			for (int i = 0; i < batchSize; i++) {
				//if ((dataIndex + i) % _steps == 0) std::cout << std::endl;

				auto input = Vector({ arguments["input"].At(i, 0) , arguments["input"].At(i, 1) });
				//auto input = arguments["input"].At(i, 0);
				auto label = arguments["label"].At(i, 0);
				auto prediction = executor->outputs[0].At(i, 0);
				auto difference = abs(label - prediction);

				auto normalizedQuantizedPrediction = inputQuantizer->QuantizeNormalizedVector(prediction);

				// Print input, label and prediction
				if ((dataIndex + i) % 250 == 0) {
					std::cout << "\ti: " << (dataIndex + i) << "\tx0: " << input[0] << "\tx1: " << input[1] << "\tl: " << label << "\tp: " << prediction << "\td: " << difference << std::endl;
					//std::cout << "\ti: " << (dataIndex + i) << "\tx0: " << input << "\tl: " << label << "\tp: " << prediction << std::endl;
				}

				// Test if correct
				if (label == normalizedQuantizedPrediction[0]) correctDataPoints++;
			}

			// Update accuracy
			auto labelShape = arguments["label"].GetShape();
			auto predShape = executor->outputs[0].GetShape();

			//accuracy.Update(arguments["label"], executor->outputs[0]);

			dataIndex += batchSize;
		}

		// Accuracy
		//std::cout << "\tMX Accuracy: " << accuracy.Get() << "%" << std::endl;

		float manualAccuracy = (float)correctDataPoints / (float)dataIndex * 100.0;
		std::cout << "\tManual accuracy: " << manualAccuracy << "%" << std::endl;

		// Make sure to delete the pointers
		delete executor;
		delete optimizer;
	}
}