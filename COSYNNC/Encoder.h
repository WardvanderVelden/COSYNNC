#pragma once
#include "FileManager.h"
#include "Quantizer.h"
#include "StringHelper.h"

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

		/// <summary>Set the threshold that determines the significance with which the neural network needs to output a value for it to be taken as the truth</summary>
		void SetThreshold(float threshold = 0.95);

		/// <summary>Set the saving path</summary>
		void SetSavingPath(string path);

		/// <summary>Train the linked neural network to encode the winning set</summary>
		/// <param name="epochs">Amount of data epochs that the neural network is trained before termination</param>
		void Train(unsigned int epochs = 10);

		/// <summary>Compute the fitness of the neural network to determine what part is currently encoded succesfully</summary>
		float ComputeFitness();

		/// <summary>Compute the amount of false positivies that the neural network provides with respect to the cardinality of the state space</summary>
		float ComputeFalsePositives();

		/// <summary>Detrain the false positives that are currently in the network</summary>
		/// <param name="epochs">Amount of epochs that the procedure should scan for false positives and detrain them</param>
		void DetrainFalsePositives(unsigned int epochs = 10);

		/// <summary>Encodes the winning set of the loaded static controller into the neural network</summary>
		/// <param name="trainingEpochs">Amount of epochs of data that the neural network trains on before checking the fitness</param>
		/// <param name="falsePositiveDetrainingEpochs">Amount of epochs of data that the neural network detrains the false positives before checking the fitness</param>
		/// <param name="passingFitness">The level of fitness that the assigned neural network should attain before the procedure. The value is a percentage e.g. 95.</param>
		/// <param name="passingFalsePositives">The level of false positives that is acceptable for termination of the encoder</param>
		void Encode(unsigned int trainingEpochs = 10, unsigned int falsePositiveDetrainingEpochs = 5, float passingFitness = 95.0, float passingFalsePositives = 0.0);
	private:
		/// <summary>Train the neural network based on the neural network inputs and assigned labels</summary>
		/// <param name="inputs">A vector of COSYNNC Vectors that represent the inputs to the neural network</param>
		/// <param name="labels">A vector of COSYNNC Vectors that represent the labels for the neural network</param>
		void TrainQueue(vector<Vector>& inputs, vector<Vector>& labels);

		FileManager _fileManager;

		Controller* _controller = nullptr;

		Quantizer* _stateQuantizer = nullptr;
		Quantizer* _inputQuantizer = nullptr;

		NeuralNetwork* _neuralNetwork = nullptr;

		// Encoder parameters
		unsigned int _batchSize = 10;
		float _threshold = 0.95; // The significance with which the neural network has to output a statement for it to be taken as the truth

		// Logging variables
		string _savingPath = "controllers";

		float _bestFitness = 0.0;
		float _bestFitnessFalsePositives = 0.0;
	};
}