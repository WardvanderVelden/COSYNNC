#include "Plant.h"

using namespace COSYNNC;

Plant::Plant(int stateSpaceDimension, int inputSpaceDimension, float tau) : _stateSpaceDim(stateSpaceDimension), _inputSpaceDim(inputSpaceDimension), _tau(tau) {
	_state.SetLength(stateSpaceDimension);
	_input.SetLength(inputSpaceDimension);
}


// Virtual function that describes the plant dynamics subject to a time step of tau, should be overriden by the actual plant dynamics
Vector Plant::StepDynamics(Vector input) {
	return _state;
}


// Defines the over approximation of the dynamics of the plant for a single time step tau
Vector Plant::StepOverApproximation(Vector input) {
	return StepDynamics(input);
}


// TODO: Turn this into a Runge-Kutta integration where the delta dynamics are described and proper numerical integration techniques are utilized
void Plant::Evolve(Vector input) {
	_state = StepDynamics(input);
	_input = input;
}


Vector Plant::GetState() const {
	return _state;
}


void Plant::SetState(Vector state) {
	_state = state;
}

void Plant::PrintState() const {
	std::cout << "Plant state:" << std::endl;
	_state.PrintValues();
}