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


	// Train the linked neural network to encode the winning set
	void Encoder::Train(unsigned int epochs) {
		const unsigned long stateSpaceCardinality = _stateQuantizer->GetCardinality();

		vector<Vector> stateQueue;
		vector<Vector> winningSetQueue;

		// Amount of epochs
		for (unsigned int epoch = 0; epoch < epochs; epoch++) {
			// Go through the static controller and add to the training queue
			for (unsigned long index = 0; index < stateSpaceCardinality; index++) {
				auto normalizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->GetVectorFromIndex(index));
				stateQueue.push_back(normalizedState);

				auto input = _controller->GetControlActionFromIndex(index);

				// Is input defined and thus in the winning set (if it was it is in the winning set as by the rules of storing static controllers)
				if (input.GetLength() > 0) winningSetQueue.push_back(Vector({ 0.0, 1.0 }));
				else winningSetQueue.push_back(Vector({ 1.0, 0.0 }));

				// Check if the queue is full, if so push to the neural network for training
				if (stateQueue.size() >= _batchSize) {
					_neuralNetwork->Train(stateQueue, winningSetQueue);

					// Clear the queue
					stateQueue.clear();
					winningSetQueue.clear();
				}
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

					if (output[0] >= 0.5 && !isInWinningSet) {
						fitIndices++;
					}
					if (output[1] >= 0.5 && isInWinningSet) {
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


	// Encodes the loaded static controller into the neural network assigned to the encoder
	void Encoder::Encode(unsigned int epochsPerTrainingSession, float passingFitness) {
		std::cout << std::endl;

		float fitness = 0.0;
		do {
			std::cout << "Training:\t";
			Train(epochsPerTrainingSession);

			fitness = ComputeFitness();
			std::cout << "\tFitness: " << fitness << std::endl;
		} while (fitness < passingFitness);

		std::cout << std::endl << "COSYNNC:\tWinning set encoded into neural network!" << std::endl;
	}
}