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

		// Get a control action based on the state in the stateSpace of the plant using a greedy strategy
		Vector GetControlAction(Vector state);

		// Get a probabilistic control action based on the certainty of the network
		Vector GetProbabilisticControlAction(Vector state, Vector& oneHot, Vector& networkOutput);

		// Returns the control specification that is currently assigned to the controller
		ControlSpecification* GetControlSpecification() const;
	private:
		float _tau;

		Vector _lastControlAction;
		//Vector _lastState;

		int _stateSpaceDim;
		int _inputSpaceDim;

		Quantizer* _stateQuantizer = NULL;
		Quantizer* _inputQuantizer = NULL;

		ControlSpecification* _controlSpecification;

		NeuralNetwork* _neuralNetwork;
	};
}


