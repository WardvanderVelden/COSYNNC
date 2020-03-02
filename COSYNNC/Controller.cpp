#include "Controller.h"

namespace COSYNNC {
	Controller::Controller() {
		_stateSpaceDim = 0;
		_inputSpaceDim = 0;

	}

	Controller::Controller(Plant* plant) {
		_stateSpaceDim = plant->GetStateSpaceDimension();
		_inputSpaceDim = plant->GetInputSpaceDimension();

		_h = plant->GetStepSize();
	}

	Controller::Controller(Plant* plant, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		_stateSpaceDim = plant->GetStateSpaceDimension();
		_inputSpaceDim = plant->GetInputSpaceDimension();

		_h = plant->GetStepSize();
		//_lastState = plant->GetState();

		_stateQuantizer = stateQuantizer;
		_inputQuantizer = inputQuantizer;
	}


	// Set the control specification of the controller
	void Controller::SetControlSpecification(ControlSpecification* specification) {
		_controlSpecification = specification;
	}


	// Set the neural network that will dictate the control input of the system
	void Controller::SetNeuralNetwork(NeuralNetwork* neuralNetwork) {
		_neuralNetwork = neuralNetwork;
	}


	// Returns a pointer to the neural network
	NeuralNetwork* Controller::GetNeuralNetwork() const {
		return _neuralNetwork;
	}


	// Read out the control action using the neural network with the quantized state as input
	Vector Controller::GetControlAction(Vector state, Vector* networkOutputVector) {
		auto quantizedNormalizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->QuantizeVector(state));
		auto networkOutput = _neuralNetwork->EvaluateNetwork(quantizedNormalizedState);

		if (networkOutputVector != NULL) *networkOutputVector = networkOutput;

		return GetControlActionFromNetwork(networkOutput);
	}


	// Get a control actions based on states in batch for computational efficiency
	Vector* Controller::GetControlActionInBatch(Vector* states, unsigned int batchSize) {
		Vector* inputs = new Vector[batchSize];
		Vector* normalizedStates = new Vector[batchSize];

		// Normalize states for neural network
		for (unsigned int i = 0; i < batchSize; i++) {
			auto state = states[i];
			normalizedStates[i] = _stateQuantizer->NormalizeVector(_stateQuantizer->QuantizeVector(state));
		}

		// Get the network outputs
		auto networkOutputs = _neuralNetwork->EvaluateNetworkInBatch(normalizedStates, batchSize);

		// Get control action from the network outputs for all outputs in batch
		for (unsigned int i = 0; i < batchSize; i++) {
			inputs[i] = GetControlActionFromNetwork(networkOutputs[i]);
		}

		delete[] normalizedStates;
		delete[] networkOutputs;

		return inputs;
	}


	// Get a probabilistic control action based on the certainty of the network
	Vector Controller::GetProbabilisticControlActionFromLabelledNeurons(Vector state, Vector& oneHot, Vector& networkOutput) {
		auto normalizedQuantizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->QuantizeVector(state));

		// Get network output for the quantized state
		networkOutput = _neuralNetwork->EvaluateNetwork(normalizedQuantizedState);

		auto amountOfOutputNeurons = networkOutput.GetLength();

		// Get total probability of all the labels
		float totalProbability = 0.0;
		for (int i = 0; i < amountOfOutputNeurons; i++) totalProbability += networkOutput[i];

		// Normalize probabilities
		for (int i = 0; i < amountOfOutputNeurons; i++) networkOutput[i] = networkOutput[i] / totalProbability;

		// Determine a one hot vector from the network output
		float randomValue = ((float)rand() / RAND_MAX);
		float cumalitiveProbability = 0.0;
		int oneHotIndex = 0;
		for (int i = 0; i < amountOfOutputNeurons; i++) {
			cumalitiveProbability += networkOutput[i];
			if (randomValue < cumalitiveProbability) {
				oneHotIndex = i;
				break;
			}
		}

		// Determine input from one hot vector
		oneHot = Vector(amountOfOutputNeurons);
		oneHot[oneHotIndex] = 1.0;

		return _inputQuantizer->GetVectorFromOneHot(oneHot);
	}


	// Get a probabilistic control action based on the certainty of the network for range neurons
	Vector Controller::GetProbabilisticControlActionFromRangeNeurons(Vector state, Vector& networkOutput) {
		const auto inputDimension = _inputQuantizer->GetSpaceDimension();

		auto normalizedQuantizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->QuantizeVector(state));

		// Get network output for the quantized state
		networkOutput = _neuralNetwork->EvaluateNetwork(normalizedQuantizedState);

		Vector normalInput(inputDimension);
		for (unsigned int i = 0; i < inputDimension; i++) {
			if (networkOutput[i * 2] < 0 || networkOutput[i * 2 + 1] < 0 || networkOutput[i * 2] > 1.0 || networkOutput[i * 2 + 1] > 1.0) {
				normalInput[i] = 0.5;
			}
			else {
				float randomValue = ((float)rand() / RAND_MAX);
				normalInput[i] = randomValue * (networkOutput[i * 2 + 1] - networkOutput[i * 2]) + networkOutput[i * 2];
			}
		}

		return _inputQuantizer->QuantizeVector(_inputQuantizer->DenormalizeVector(normalInput));
	}


	// Processes the network output to get the input based on the output type
	Vector Controller::GetControlActionFromNetwork(Vector networkOutput) {
		auto neuralNetworkOutputType = _neuralNetwork->GetOutputType();

		switch (neuralNetworkOutputType) {
			case OutputType::Labelled: {
				return GetGreedyInputFromLabelledNeurons(networkOutput);
				break;
			}
			case OutputType::Range: {
				return GetGreedyInputFromRangeNeurons(networkOutput);
				break;
			}
		}

		return Vector();
	}


	// Processes network output for labelled neurons to get an input greedily
	Vector Controller::GetGreedyInputFromLabelledNeurons(Vector networkOutput) {
		auto networkOutputLength = networkOutput.GetLength();

		float highestProbability = 0.0;
		int highestProbabilityIndex = 0.0;
		for (int i = 0; i < networkOutputLength; i++) {
			if (networkOutput[i] > highestProbability) {
				highestProbability = networkOutput[i];
				highestProbabilityIndex = i;
			}
		}

		Vector oneHot(networkOutputLength);
		oneHot[highestProbabilityIndex] = 1.0;

		return _inputQuantizer->GetVectorFromOneHot(oneHot);
	}


	// Processes network output for range neurosn to get an input greedily
	Vector Controller::GetGreedyInputFromRangeNeurons(Vector networkOutput) {
		auto inputDimension = _inputQuantizer->GetSpaceDimension();

		Vector normalInput(inputDimension);
		for (unsigned int i = 0; i < inputDimension; i++) {
			if (networkOutput[i * 2] < 0 || networkOutput[i * 2 + 1] < 0 || networkOutput[i * 2] > 1.0 || networkOutput[i * 2 + 1] > 1.0) {
				normalInput[i] = 0.5;
			}
			else {
				normalInput[i] = (max(networkOutput[i * 2], networkOutput[i * 2 + 1]) - min(networkOutput[i * 2], networkOutput[i * 2 + 1])) / 2 + min(networkOutput[i * 2], networkOutput[i * 2 + 1]);
			}
		}

		auto denormalInput = _inputQuantizer->DenormalizeVector(normalInput);
		return _inputQuantizer->QuantizeVector(denormalInput);
	}


	// Returns the control specification that is currently assigned to the controller
	ControlSpecification* Controller::GetControlSpecification() const {
		return _controlSpecification;
	}
}