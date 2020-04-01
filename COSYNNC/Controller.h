#pragma once
#include "Vector.h";
#include "Plant.h"
#include "Quantizer.h"
#include "ControlSpecification.h"
#include <math.h>
#include "NeuralNetwork.h"

namespace COSYNNC {
	class Controller
	{
	public:
		// Initializes the controller
		Controller();

		// Initializes the controller based on the plant
		Controller(Plant* plant);

		// Initialize the controller based on the plant and both the state and input quantizers
		Controller(Plant* plant, Quantizer* stateQuantizer, Quantizer* inputQuantizer);

		// Set the control specification of the controller
		void SetControlSpecification(ControlSpecification* specification);

		// Set the neural network that will dictate the control input of the system
		void SetNeuralNetwork(NeuralNetwork* neuralNetwork);

		// Returns a pointer to the neural network
		NeuralNetwork* GetNeuralNetwork() const;

		// Get a control action based on the state in the stateSpace of the plant using a greedy strategy
		Vector GetControlAction(Vector state, Vector* networkOutputVector = NULL);

		// Get a control actions based on states in batch for computational efficiency
		Vector* GetControlActionInBatch(Vector* states, unsigned int batchSize);

		// Get a probabilistic control action based on the certainty of the network for labelled neurons
		Vector GetProbabilisticControlActionFromLabelledNeurons(Vector state, Vector& oneHot, Vector& networkOutput);

		// Get a probabilistic control action based on the certainty of the network for range neurons
		Vector GetProbabilisticControlActionFromRangeNeurons(Vector state, Vector& networkOutput);

		// Processes the network output to get the input based on the output type
		Vector GetControlActionFromNetwork(Vector networkOutput);

		// Processes network output for labelled neurons to get an input greedily
		Vector GetGreedyInputFromLabelledNeurons(Vector networkOutput);

		// Processes network output for range neurosn to get an input greedily
		Vector GetGreedyInputFromRangeNeurons(Vector networkOutput);

		// Returns the control specification that is currently assigned to the controller
		ControlSpecification* GetControlSpecification() const;

		// Compile the inputs array that states the input for every index in the state space
		void CompileInputArray();

		// Get input from the input array based on the index
		Vector GetControlActionFromIndex(long index) const;
	private:
		float _h;

		Vector _lastControlAction;
		//Vector _lastState;

		int _stateSpaceDim;
		int _inputSpaceDim;

		Quantizer* _stateQuantizer = nullptr;
		Quantizer* _inputQuantizer = nullptr;

		ControlSpecification* _controlSpecification = nullptr;

		NeuralNetwork* _neuralNetwork = nullptr;

		Vector* _inputArray = nullptr;
	};
}


