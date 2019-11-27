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
				_outputs[i] = Activation(fullyConnected, ActivationActType::kRelu); // Dont use sigmoid, its slow and shit
			}
		}

		_network = LinearRegressionOutput(_outputs.back(), label);
	}

	// DEBUG: Temporay test bed for learning MXNET
	void MultilayerPerceptron::Test(TrainingData* data, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		const int batchSize = 250;
		const float learningRate = 0.1;
		const float weightDecay = 0.01;
		const int maxEpoch = 25000;
		const int dataSize = data->labels.Size()/2;

		// Arguments for the neural network (input layer is 2 neurons and the output layer is 1 neuron)
		map<string, NDArray> arguments; 
		arguments["input"] = NDArray(Shape(batchSize, 1), _context);
		arguments["label"] = NDArray(Shape(batchSize, 2), _context);

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
		std::cout << "Training network" << std::endl;

		int currentDataIndex = 0;
		int sampled = 0;
		int correct = 0;
		for (int epoch = 0; epoch < maxEpoch; ++epoch) {
			// Reset data
			currentDataIndex = 0;

			// While still have data
			while (currentDataIndex <= (dataSize - batchSize)) {			
				// Get data batch
				NDArray inputBatch = data->inputs.Slice(currentDataIndex, currentDataIndex + batchSize);
				NDArray labelBatch = data->labels.Slice(currentDataIndex, currentDataIndex + batchSize);

				// Copy data to the network
				inputBatch.CopyTo(&arguments["input"]);
				labelBatch.CopyTo(&arguments["label"]);

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
				if (epoch % 100 == 0 && currentDataIndex == 0) {
					std::cout << "Epoch: " << epoch << std::endl;

					int randomIndex = rand() % batchSize;
					Vector label = Vector({ arguments["label"].At(randomIndex, 0), arguments["label"].At(randomIndex, 1) });
					Vector prediction = Vector({ executor->outputs[0].At(randomIndex, 0), executor->outputs[0].At(randomIndex, 1) });

					sampled++;
					if (round(label[0]) == round(prediction[0]) && round(label[1]) == round(prediction[1])) {
						correct++;
					}
					float accuracy = (float)correct / (float)sampled * 100.0;

					std::cout << "\tx0: " << arguments["input"].At(randomIndex, 0) << "\tl0: " << label[0] << "\tl1: " << label[1] << "\tp0: " << prediction[0] << "\tp1: " << prediction[1] << "\tsa: " << accuracy << "%" << std::endl;
				}

				currentDataIndex += batchSize;
			}
		}
		std::cout << std::endl << "Stopped training!" << std::endl << std::endl;

		// Evaluate network
		std::cout << "Validating network" << std::endl;
		//Accuracy accuracy;
		correct = 0;
		sampled = 0;

		currentDataIndex = 0;
		while (currentDataIndex <= (dataSize - batchSize)) {
			// Get data batch
			NDArray inputBatch = data->inputs.Slice(currentDataIndex, currentDataIndex + batchSize);
			NDArray labelBatch = data->labels.Slice(currentDataIndex, currentDataIndex + batchSize);

			auto inputBatchShape = inputBatch.GetShape();
			auto labelBatchShape = labelBatch.GetShape();

			// Copy data to the network
			inputBatch.CopyTo(&arguments["input"]);
			labelBatch.CopyTo(&arguments["label"]);

			// Execute the network
			executor->Forward(false);

			// Determine accuracy
			for (int i = 0; i < batchSize; i++) {
				Vector label = Vector({ arguments["label"].At(i, 0), arguments["label"].At(i, 1) });
				Vector prediction = Vector({ executor->outputs[0].At(i, 0), executor->outputs[0].At(i, 1) });

				if ((currentDataIndex + i) % _steps == 0) std::cout << std::endl;

				std::cout << "i: " << (currentDataIndex + i) << "\tx0: " << arguments["input"].At(i, 0) << "\tl0: " << label[0] << "\tl1: " << label[1] << "\tp0: " << prediction[0] << "\tp1: " << prediction[1] << std::endl;

				// Test if correct
				sampled++;
				if (round(label[0]) == round(prediction[0]) && label[1] == round(prediction[1])) correct++;
			}
			currentDataIndex += batchSize;
		}

		// Accuracy
		float manualAccuracy = (float)correct / (float)sampled * 100.0;
		//std::cout << "Accuracy: " << accuracy.Get() << "%" <<  std::endl;
		std::cout << "Manual accuracy: " << manualAccuracy << "%" << std::endl;

		// Make sure to delete the pointers
		delete executor;
		delete optimizer;
	}

	TrainingData* MultilayerPerceptron::GetTrainingData(Plant* plant, Controller* controller, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		const int episodes = 500;

		vector<mx_float> inputs;
		vector<mx_float> labels;

		// Simulate episodes
		for (int i = 0; i < episodes; i++) {
			// Set initial position
			controller->ResetController();
			plant->SetState(stateQuantizer->GetRandomVector());

			for (int j = 0; j < _steps; j++) {
				Vector quantizedState = stateQuantizer->QuantizeVector(plant->GetState());
				Vector normalizedQuantizedState = stateQuantizer->NormalizeVector(quantizedState);

				Vector controlAction = controller->GetPDControlAction(quantizedState);
				Vector quantizedControlAction = inputQuantizer->QuantizeVector(controlAction);

				Vector normalizedQuantizedControlAction = inputQuantizer->NormalizeVector(quantizedControlAction);

				// Add data 
				/*inputs.push_back(normalizedQuantizedState[0]);
				inputs.push_back(normalizedQuantizedState[1]);

				//labels.push_back(normalizedQuantizedControlAction[0]);

				// Softmax model for testing
				if (normalizedQuantizedControlAction[0] > 0.5) {
					labels.push_back(0.0);
					labels.push_back(1.0);
				}
				else {
					labels.push_back(1.0);
					labels.push_back(0.0);
				}*/

				// Simple single input single output model
				mx_float randomValue = (rand() % 100000) / 100000.0 * 3.0;
				inputs.push_back(randomValue);
				
				if (randomValue > 0.5) {
					labels.push_back(0.0);
					labels.push_back(1.0);
				}
				else {
					labels.push_back(1.0);
					labels.push_back(0.0);
				}

				// Evolve plant with control action
				plant->Evolve(quantizedControlAction);
			}
		}

		TrainingData* data = new TrainingData();
		/*data->inputs = NDArray(inputs, Shape(episodes*_steps, 2), _context);
		data->labels = NDArray(labels, Shape(episodes*_steps, 2), _context);*/
		data->inputs = NDArray(inputs, Shape(episodes*_steps, 1), _context);
		data->labels = NDArray(labels, Shape(episodes*_steps, 2), _context);

		return data;
	}
}