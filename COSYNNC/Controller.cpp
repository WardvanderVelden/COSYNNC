#include "Controller.h"

namespace COSYNNC {
	Controller::Controller() {
		_stateSpaceDim = 0;
		_inputSpaceDim = 0;

		_tau = 0.0;
	}

	Controller::Controller(Plant* plant) {
		_stateSpaceDim = plant->GetStateSpaceDimension();
		_inputSpaceDim = plant->GetInputSpaceDimension();

		_tau = plant->GetTau();
	}

	Controller::Controller(Plant* plant, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		_stateSpaceDim = plant->GetStateSpaceDimension();
		_inputSpaceDim = plant->GetInputSpaceDimension();

		_tau = plant->GetTau();
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
		if (_neuralNetwork == NULL) return Vector(_inputSpaceDim);

		auto quantizedNormalizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->QuantizeVector(state));
		auto networkOutput = _neuralNetwork->EvaluateNetwork(quantizedNormalizedState);

		// Greedily pick the input
		float highestProbability = 0.0;
		int highestProbabilityIndex = 0.0;
		for (int i = 0; i < networkOutput.GetLength(); i++) {
			if (networkOutput[i] > highestProbability) {
				highestProbability = networkOutput[i];
				highestProbabilityIndex = i;
			}
		}

		Vector oneHot(networkOutput.GetLength());
		oneHot[highestProbabilityIndex] = 1.0;

		auto input = _inputQuantizer->GetVectorFromOneHot(oneHot);

		if (networkOutputVector != NULL) *networkOutputVector = networkOutput;

		return input;
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

		for (unsigned int i = 0; i < batchSize; i++) {
			auto networkOutput = networkOutputs[i];

			// Greedily pick the input
			float highestProbability = 0.0;
			int highestProbabilityIndex = 0.0;
			for (int j = 0; j < networkOutput.GetLength(); j++) {
				if (networkOutput[j] > highestProbability) {
					highestProbability = networkOutput[j];
					highestProbabilityIndex = j;
				}
			}

			Vector oneHot(networkOutput.GetLength());
			oneHot[highestProbabilityIndex] = 1.0;

			inputs[i] = _inputQuantizer->GetVectorFromOneHot(oneHot);
		}

		delete[] normalizedStates;
		delete[] networkOutputs;

		return inputs;
	}


	// Get a probabilistic control action based on the certainty of the network
	Vector Controller::GetProbabilisticControlAction(Vector state, Vector& oneHot, Vector& networkOutput) {
		auto normalizedQuantizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->QuantizeVector(state));

		// Get network output for the quantized state
		networkOutput = _neuralNetwork->EvaluateNetwork(normalizedQuantizedState);
		//if (j == 0) networkOutput = _neuralNetwork->EvaluateNetwork(normalizedQuantizedState); // TODO: Find a more expedient solution to this bug

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


	// Returns the control specification that is currently assigned to the controller
	ControlSpecification* Controller::GetControlSpecification() const {
		return _controlSpecification;
	}
}