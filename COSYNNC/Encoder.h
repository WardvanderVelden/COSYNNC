#pragma once
#include "FileManager.h"
#include "Quantizer.h"

namespace COSYNNC {
	class Encoder {
	public:
		/// <summary>Initializes the encoder based on a SCOTS style static controller</summary>
		Encoder(string path, string name);

		/// <summary>Initializes the encoder based on an abstraction</summary>
		Encoder(Abstraction abstraction);

		/// <summary>Default destructor</summary>
		~Encoder();

		/// <summary>Set the neural network</summary>
		void SetNeuralNetwork(NeuralNetwork* neuralNetwork);

		/// <summary>Set the batch size of the encoder during training</summary>
		void SetBatchSize(unsigned int batchSize = 10);

		/// <summary>Train the linked neural network to encode the winning set</summary>
		/// <param name="epochs">Amount of data epochs that the neural network is trained before termination</param>
		void Train(unsigned int epochs = 10);

		/// <summary>Compute the fitness of the neural network to determine what part is currently encoded succesfully</summary>
		float ComputeFitness();

		/// <summary>Encodes the winning set of the loaded static controller into the neural network</summary>
		/// <param name="epochsPerTrainingSession">Amount of epochs of data that the neural network trains on before checking the fitness</param>
		/// <param name="passingFitness">The level of fitness that the assigned neural network should attain before the procedure. The value is a percentage e.g. 95.</param>
		void Encode(unsigned int epochsPerTrainingSession = 10, float passingFitness = 95.0);
	private:
		FileManager _fileManager;

		Controller* _controller = nullptr;

		Quantizer* _stateQuantizer = nullptr;
		Quantizer* _inputQuantizer = nullptr;

		NeuralNetwork* _neuralNetwork = nullptr;

		// Encoder parameters
		unsigned int _batchSize = 10;
	};
}