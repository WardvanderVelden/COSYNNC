#include "Controller.h"

using namespace COSYNNC;

Controller::Controller() {
	_stateSpaceDim = 0;
	_inputSpaceDim = 0;
}

Controller::Controller(Plant* plant) {
	_stateSpaceDim = plant->GetStateSpaceDimension();
	_inputSpaceDim = plant->GetInputSpaceDimension();
}

Controller::Controller(Plant* plant, Quantizer* quantizer) {
	_stateSpaceDim = plant->GetStateSpaceDimension();
	_inputSpaceDim = plant->GetInputSpaceDimension();

	_quantizer = quantizer;
}


// Read out the control action using the neural network with the quantized state as input
Vector Controller::GetControlAction(Vector state) {
	Vector controlAction(1);
	controlAction[0] = 1000;

	return controlAction;
}
