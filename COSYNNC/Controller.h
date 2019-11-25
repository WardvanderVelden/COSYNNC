#pragma once
#include "Vector.h";
#include "Plant.h"
#include "Quantizer.h"
#include "ControlSpecification.h"
#include <math.h>

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

		// Get a control action based on the state in the stateSpace of the plant
		Vector GetControlAction(Vector state);

		// DEBUG: Temporay PD controller in order to have some sort of benchmark of data generator
		Vector GetPDControlAction(Vector state);

		// DEBUG: Resets the control so no old information is used for calcuating the input
		void ResetController();
	private:
		float _tau;

		Vector _lastControlAction;
		Vector _lastState;

		int _stateSpaceDim;
		int _inputSpaceDim;

		Quantizer* _stateQuantizer = NULL;
		Quantizer* _inputQuantizer = NULL;

		ControlSpecification* _controlSpecification;
	};
}


