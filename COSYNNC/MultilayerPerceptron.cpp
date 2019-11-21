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
				_outputs[i] = Activation(fullyConnected, ActivationActType::kSigmoid);
			}
		}

		// DEBUG: This is now a softmax function (so cross entropy) just to test all the neural network functionality
		_network = SoftmaxOutput(_outputs.back(), label);
	}

	// DEBUG: Temporay test bed for learning MXNET
	void MultilayerPerceptron::Test(TrainingData* data) {
		const int batchSize = 50;
		const float learningRate = 0.1;
		const float weightDecay = 0.1;
		const int maxEpoch = 100;
		const int dataSize = data->labels.Size();

		// Arguments for the neural network (input layer is 2 neurons and the output layer is 1 neuron)
		map<string, NDArray> arguments; 
		arguments["input"] = NDArray(Shape(batchSize, 1), _context);
		arguments["output"] = NDArray(Shape(batchSize, 1), _context);

		// Infers the matrix sizes for the network from the netwokr arguments
		_network.InferArgsMap(_context, &arguments, arguments);

		// Initialize all parameters with a uniform distribution
		auto initializer = Uniform(-0.1, 0.1);
		for (auto& argument : arguments) {
			initializer(argument.first, &argument.second);
		}

		// Create an optimizer (for now we will use simply stochastic gradient descent (sgd))
		Optimizer* optimizer = OptimizerRegistry::Find("sgd");
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
			while (currentDataIndex < (dataSize - batchSize)) {			
				// Get data batch
				NDArray inputBatch = data->inputs.Slice(currentDataIndex, currentDataIndex + batchSize);
				NDArray labelBatch = data->labels.Slice(currentDataIndex, currentDataIndex + batchSize);

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
		std::cout << std::endl << "Stopped training!" << std::endl;

		// Evaluate network
		/*Accuracy accuracy;

		currentDataIndex = 0;
		while (currentDataIndex < (dataSize - batchSize)) {
			// Get data batch
			NDArray inputBatch = data->inputs.Slice(currentDataIndex, currentDataIndex + batchSize);
			NDArray labelBatch = data->labels.Slice(currentDataIndex, currentDataIndex + batchSize);

			// Copy data to the network
			inputBatch.CopyTo(&arguments["input"]);
			labelBatch.CopyTo(&arguments["output"]);

			// Execute the network
			executor->Forward(false);

			// DEBUG
			auto shape = executor->outputs[0].GetShape();
			auto value = executor->outputs[0].GetData();
			auto label = labelBatch.GetData();

			// Update accuracy
			accuracy.Update(labelBatch, executor->outputs[0]);

			currentDataIndex += batchSize;
		}
		std::cout << "Accuracy: " << accuracy.Get() << std::endl;*/

		// Manually evaluate the network
		arguments["input"] = NDArray(Shape(1, 1), _context);
		arguments["output"] = NDArray(Shape(1, 1), _context);

		currentDataIndex = 0;
		int correct = 0;
		int total = 0;
		while (currentDataIndex < dataSize) {
			NDArray inputBatch = data->inputs.Slice(currentDataIndex, currentDataIndex + 1);
			NDArray labelBatch = data->labels.Slice(currentDataIndex, currentDataIndex + 1);

			inputBatch.CopyTo(&arguments["input"]);
			labelBatch.CopyTo(&arguments["output"]);

			// Execute the network
			executor->Forward(false);

			auto label = labelBatch.GetData();
			auto prediction = executor->outputs[0].GetData();

			if (*label == 1.0 && *prediction > 0.5) correct++;
			if (*label == 0.0 && *prediction < 0.5) correct++;

			if (isnan(*prediction)) total++;

			currentDataIndex++;
		}
		float accuracy = (float)correct / (float)total * 100.0;
		std::cout << "Accuracy: " << accuracy << "%" << std::endl;

		// Make sure to delete the pointers
		delete executor;
		delete optimizer;
	}

	TrainingData* MultilayerPerceptron::GetTrainingData(Plant* plant, Controller* controller, Quantizer* stateQuantizer) {
		const int episodes = 1000;
		const int steps = 50;

		//NDArray input(Shape(episodes*steps, 2), _context);
		//NDArray label(Shape(episodes*steps, 1), _context);

		vector<mx_float> input;
		vector<mx_float> label;

		// Simulate a 1000 episodes
		for (int i = 0; i < episodes; i++) {
			// Set initial position
			/*plant->SetState(stateQuantizer->GetRandomVector());

			for (int j = 0; j < steps; j++) {
				Vector state = stateQuantizer->QuantizeVector(plant->GetState());
				Vector controlAction = controller->GetPDControlAction(state);
				plant->Evolve(controlAction);

				// Add data 
				input.push_back(state[0]*1/10);
				input.push_back(state[1]*1/10);

				if (controlAction[0] > 2750) {
					label.push_back(1);
				}
				else {
					label.push_back(0);
				}
			}*/

			// Round function
			auto number = rand();
			input.push_back(number);

			if (number > 0.5) label.push_back(1);
			else label.push_back(0);
		}

		TrainingData* data = new TrainingData();
		data->inputs = NDArray(input, Shape(episodes*steps, 1), _context);
		data->labels = NDArray(label, Shape(episodes*steps, 1), _context);
		return data;
	}
}