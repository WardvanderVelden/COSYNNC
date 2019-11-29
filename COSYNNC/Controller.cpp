#include "Controller.h"

using namespace COSYNNC;

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


void Controller::SetControlSpecification(ControlSpecification* specification) {
	_controlSpecification = specification;
}


void Controller::SetNeuralNetwork(NeuralNetwork* neuralNetwork) {
	_neuralNetwork = neuralNetwork;
}

// Read out the control action using the neural network with the quantized state as input
Vector Controller::GetControlAction(Vector state) {
	if (_neuralNetwork == NULL) return Vector(_inputSpaceDim);

	auto quantizedNormalizedState = _stateQuantizer->NormalizeVector(_stateQuantizer->QuantizeVector(state));
	auto networkOutput = _neuralNetwork->EvaluateNetwork(quantizedNormalizedState);
	
	auto denormalizeddInput = _inputQuantizer->DenormalizeVector(networkOutput);
	auto quantizedInput = _inputQuantizer->QuantizeVector(denormalizeddInput);
	
	return quantizedInput;
}

// DEBUG: Temporay PD controller in order to have some sort of benchmark of data generator
/*Vector Controller::GetPDControlAction(Vector state) {
	Vector goal = _controlSpecification->GetCenter();

	// Calculate proportional and derivative
	Vector proportional = (Vector(goal) - state);
	Vector lastProportional = (Vector(goal) - _lastState);

	Vector derivative = (proportional - lastProportional) / _tau;

	Vector controlAction(_inputSpaceDim);

	// Determine control action based on two PD controllers and where we are in the state space
	if (state[0] > goal[0]) controlAction[0] = 2500 + proportional[0] * 500 + derivative[1] * 100;
	else controlAction[0] = 2500 + proportional[0] * 2500 + derivative[1] * 100;

	if (controlAction[0] < 0) controlAction[0] = 0;

	return controlAction;
}

void Controller::ResetController() {
	_lastState = Vector(_stateSpaceDim);
}*/