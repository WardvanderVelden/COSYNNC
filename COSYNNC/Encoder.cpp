#include "Encoder.h"

namespace COSYNNC {
	// Initializes the encoder based on a SCOTS style static controller
	Encoder::Encoder(string path, string name) {
		_fileManager = FileManager();

		_controller = _fileManager.LoadStaticController(path, name);

		_stateQuantizer = _controller->GetStateQuantizer();
		_inputQuantizer = _controller->GetInputQuantizer();

		std::cout << "COSYNNC:\tEncoding the winning set into a neural network" << std::endl;
	}


	// Initializes the encoder based on an abstraction
	Encoder::Encoder(Abstraction abstraction) {
		_stateQuantizer = abstraction.GetStateQuantizer();
		_inputQuantizer = abstraction.GetInputQuantizer();
	}


	// Default destructor
	Encoder::~Encoder() {
		delete _stateQuantizer;
		delete _inputQuantizer;

		delete _controller;
	}


	// Set the neural network
	void Encoder::SetNeuralNetwork(NeuralNetwork* neuralNetwork) {
		_neuralNetwork = neuralNetwork;

		_neuralNetwork->ConfigureInputOutput(_stateQuantizer->GetDimension(), 2, _batchSize, 0.1);

		auto dataSize = _neuralNetwork->GetDataSize();
		std::cout << "COSYNNC:\tNeural network has data size: " + std::to_string(dataSize) + " bytes" << std::endl;
	}


	// Set the batch size during training
	void Encoder::SetBatchSize(unsigned int batchSize) {
		_batchSize = batchSize;
	}


	// Set the threshold that determines the significance with which the neural network needs to output a value for it to be taken as the truth</summary>
	void Encoder::SetThreshold(float threshold) {
		_threshold = threshold;
	}


	// Set the saving path
	void Encoder::SetSavingPath(string path) {
		_savingPath = path;
	}


	// Train the linked neural network to encode the winning set
	void Encoder::Train(unsigned int epochs) {
		const unsigned long stateSpaceCardinality = _stateQuantizer->GetCardinality();

		vector<Vector> states;
		vector<Vector> labels;

		// Amount of epochs
		for (unsigned int epoch = 0; epoch < epochs; epoch++) {
			// Go through the static controller and add to the training queue
			for (unsigned long index = 0; index < stateSpaceCardinality; index++) {
				auto normalizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->GetVectorFromIndex(index));
				states.push_back(normalizedState);

				auto input = _controller->GetControlActionFromIndex(index);

				// Is input defined and thus in the winning set (if it was it is in the winning set as by the rules of storing static controllers)
				if (input.GetLength() > 0) labels.push_back(Vector({ 0.0, 1.0 }));
				else labels.push_back(Vector({ 1.0, 0.0 }));

				// Check if the queue is full, if so push to the neural network for training
				/*if (states.size() >= _batchSize) {
					_neuralNetwork->Train(states, labels);

					// Clear the queue
					states.clear();
					labels.clear();
				}*/
				TrainQueue(states, labels);
			}
			std::cout << ".";
		}
	}


	// Compute the fitness of the neural network to determine what part is currently encoded succesfully
	float Encoder::ComputeFitness() {
		const unsigned long stateSpaceCardinality = _stateQuantizer->GetCardinality();
		unsigned long fitIndices = 0;

		vector<Vector> stateQueue;
		vector<bool> inWinningSet;

		for (unsigned long index = 0; index < stateSpaceCardinality; index++) {
			auto normalizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->GetVectorFromIndex(index));
			stateQueue.push_back(normalizedState);

			auto input = _controller->GetControlActionFromIndex(index);

			if (input.GetLength() > 0) inWinningSet.push_back(true);
			else inWinningSet.push_back(false);

			// If the state queue is full or if it is the last index
			if (stateQueue.size() >= _batchSize || index == (stateSpaceCardinality - 1)) {
				auto outputs = _neuralNetwork->EvaluateNetworkInBatch(stateQueue);

				for (unsigned int i = 0; i < stateQueue.size(); i++) {
					auto output = outputs[i];
					bool isInWinningSet = inWinningSet[i];

					if (output[0] >= _threshold && !isInWinningSet) {
						fitIndices++;
					}
					if (output[1] >= _threshold && isInWinningSet) {
						fitIndices++;
					}
				}

				delete[] outputs;

				// Clear the queue
				stateQueue.clear();
				inWinningSet.clear();
			}
		}

		return fitIndices / (float)stateSpaceCardinality * 100.0;
	}


	//Compute the amount of false positivies that the neural network provides with respect to the cardinality of the state space</summary>
	float Encoder::ComputeFalsePositives() {
		const unsigned long stateSpaceCardinality = _stateQuantizer->GetCardinality();
		unsigned long falsePositiveIndices = 0;

		vector<Vector> stateQueue;
		vector<bool> inWinningSet;

		for (unsigned long index = 0; index < stateSpaceCardinality; index++) {
			auto normalizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->GetVectorFromIndex(index));
			stateQueue.push_back(normalizedState);

			auto input = _controller->GetControlActionFromIndex(index);

			if (input.GetLength() > 0) inWinningSet.push_back(true);
			else inWinningSet.push_back(false);

			// If the state queue is full or if it is the last index
			if (stateQueue.size() >= _batchSize || index == (stateSpaceCardinality - 1)) {
				auto outputs = _neuralNetwork->EvaluateNetworkInBatch(stateQueue);

				for (unsigned int i = 0; i < stateQueue.size(); i++) {
					auto output = outputs[i];
					bool isInWinningSet = inWinningSet[i];

					if (output[1] >= _threshold && !isInWinningSet) {
						falsePositiveIndices++;
					}
				}

				delete[] outputs;

				// Clear the queue
				stateQueue.clear();
				inWinningSet.clear();
			}
		}

		return falsePositiveIndices / (float)stateSpaceCardinality * 100.0;;
	}


	// Detrain the false positives that are currently in the network
	void Encoder::DetrainFalsePositives(unsigned int epochs) {
		const unsigned long stateSpaceCardinality = _stateQuantizer->GetCardinality();

		vector<Vector> stateQueue;
		vector<bool> inWinningSet;

		vector<Vector> falsePositiveStateQueue;
		vector<Vector> falsePositiveLabels;

		for (unsigned int epoch = 0; epoch < epochs; epoch++) {
			for (unsigned long index = 0; index < stateSpaceCardinality; index++) {
				auto normalizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->GetVectorFromIndex(index));
				stateQueue.push_back(normalizedState);

				auto input = _controller->GetControlActionFromIndex(index);

				if (input.GetLength() > 0) inWinningSet.push_back(true);
				else inWinningSet.push_back(false);

				// If the state queue is full or if it is the last index
				if (stateQueue.size() >= _batchSize || index == (stateSpaceCardinality - 1)) {
					auto outputs = _neuralNetwork->EvaluateNetworkInBatch(stateQueue);

					for (unsigned int i = 0; i < stateQueue.size(); i++) {
						auto output = outputs[i];
						bool isInWinningSet = inWinningSet[i];

						if (output[1] >= _threshold && !isInWinningSet) {
							falsePositiveStateQueue.push_back(stateQueue[i]);
							falsePositiveLabels.push_back(Vector({ 1.0, 0.0 }));

							TrainQueue(falsePositiveStateQueue, falsePositiveLabels);
						}
					}

					delete[] outputs;

					// Clear the queue
					stateQueue.clear();
					inWinningSet.clear();
				}
			}
			std::cout << "+";
		}
	}


	// Encodes the loaded static controller into the neural network assigned to the encoder
	void Encoder::Encode(unsigned int trainingEpochs, unsigned int falsePositiveDetrainingEpochs, float passingFitness, float passingFalsePositives) {
		std::cout << std::endl;

		StringHelper stringHelper;

		float fitness = 0.0;
		float falsePositives = 0.0;
		do {
			std::cout << "Training: ";
			Train(trainingEpochs);
			DetrainFalsePositives(falsePositiveDetrainingEpochs);

			fitness = ComputeFitness();
			falsePositives = ComputeFalsePositives();

			if (fitness > _bestFitness && falsePositives <= passingFalsePositives) {
				_bestFitness = fitness;
				_bestFitnessFalsePositives = falsePositives;

				string fitnessString = to_string(_bestFitness);
				stringHelper.ReplaceAll(fitnessString, '.','-');

				_fileManager.SaveNetworkAsMATLAB(_savingPath + "/winningsets", "winningset" + fitnessString, _neuralNetwork, _controller);
			}

			std::cout << "\tFitness: " << fitness << "\tFalse positives: " << falsePositives << "\tBest: " << _bestFitness << " - " << _bestFitnessFalsePositives << std::endl;
		} while (fitness < passingFitness || falsePositives > passingFalsePositives);

		std::cout << std::endl << "COSYNNC:\tWinning set encoded into neural network!" << std::endl;
	}


	// Train the neural network based on the neural network inputs and assigned labels
	void Encoder::TrainQueue(vector<Vector>& inputs, vector<Vector>& labels) {
		// Check if the queue is full, if so push to the neural network for training
		if (inputs.size() >= _batchSize) {
			_neuralNetwork->Train(inputs, labels);

			// Clear the queue
			inputs.clear();
			labels.clear();
		}
	}
}