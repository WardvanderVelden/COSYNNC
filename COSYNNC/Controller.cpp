#include "Controller.h"

namespace COSYNNC {
	// Initializes the controller
	Controller::Controller() {
		_stateSpaceDim = 0;
		_inputSpaceDim = 0;

	}


	// Initializes the controller based on the plant
	Controller::Controller(Plant* plant) {
		_stateSpaceDim = plant->GetStateSpaceDimension();
		_inputSpaceDim = plant->GetInputSpaceDimension();

		_h = plant->GetStepSize();
	}


	// Initialize the controller based on the plant and both the state and input quantizers
	Controller::Controller(Plant* plant, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		_stateSpaceDim = plant->GetStateSpaceDimension();
		_inputSpaceDim = plant->GetInputSpaceDimension();

		_h = plant->GetStepSize();
		//_lastState = plant->GetState();

		_stateQuantizer = stateQuantizer;
		_inputQuantizer = inputQuantizer;
	}


	// Initialize the controller based on the quantizers to allow for a controller without knowledge of the plant
	Controller::Controller(Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		_stateQuantizer = stateQuantizer;
		_inputQuantizer = inputQuantizer;
	}


	// Default destructor
	Controller::~Controller() {
		delete[] _inputs;
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


	// Returns a pointer to the state quantizer
	Quantizer* Controller::GetStateQuantizer() const {
		return _stateQuantizer;
	}

	// Returns a pointer to the state quantizer
	Quantizer* Controller::GetInputQuantizer() const {
		return _inputQuantizer;
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
		const auto inputDimension = _inputQuantizer->GetDimension();

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
		auto inputDimension = _inputQuantizer->GetDimension();

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


	// Compile the inputs array that states the input for every index in the state space
	void Controller::ComputeInputs() {
		delete[] _inputs;

		// Setup input array
		const auto spaceCardinality = _stateQuantizer->GetCardinality();
		_inputs = new Vector[spaceCardinality];

		// Calculate amount of batches required
		const auto batchSize = GetNeuralNetwork()->GetBatchSize();
		const long amountOfBatches = ceil(spaceCardinality / batchSize);

		// Go through all the indices through batches
		for (unsigned long batch = 0; batch <= amountOfBatches; batch++) {
			unsigned long indexOffset = batch * batchSize;
			unsigned int currentBatchSize = batchSize;
			if (batch == amountOfBatches) {
				currentBatchSize = spaceCardinality - indexOffset;

				if (currentBatchSize == 0) break;
			}

			// Collect all the states in the current batch
			Vector* states = new Vector[currentBatchSize];

			for (unsigned int i = 0; i < currentBatchSize; i++) {
				long index = indexOffset + i;
				states[i] = _stateQuantizer->GetVectorFromIndex(index);
			}

			// Get the corresponding inputs through batch network evaluation
			Vector* inputs = new Vector[currentBatchSize];
			inputs = GetControlActionInBatch(states, currentBatchSize);

			// Compute the transition function for all the states in the batch
			for (unsigned int i = 0; i < currentBatchSize; i++) {
				long index = indexOffset + i;

				_inputs[index] = inputs[i];
			}

			delete[] inputs;
			delete[] states;
		}
	}


	// Initialize the array of precomputed inputs
	void Controller::InitializeInputs() {
		delete[] _inputs;

		const auto spaceCardinality = _stateQuantizer->GetCardinality();
		_inputs = new Vector[spaceCardinality];

		for (unsigned long i = 0; i < spaceCardinality; i++) _inputs[i] = Vector((unsigned int)0);
	}


	// Sets the input for a given state in the state space
	void Controller::SetInput(unsigned long stateIndex, Vector input) {
		_inputs[stateIndex] = input;
	}


	// Get input from the input array based on the index
	Vector Controller::GetControlActionFromIndex(long index) const {
		return _inputs[index];
	}
}