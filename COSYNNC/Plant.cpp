#include "Plant.h"

using namespace cosynnc;

Plant::Plant(int stateDimension, int inputDimension, float tau) : _stateDim(stateDimension), _inputDim(inputDimension), _tau(tau) {
	_state.SetLength(stateDimension);
	_input.SetLength(inputDimension);
}


// Virtual Plant Dynamics function, should be overriden by the actual plant dynamics
Vector Plant::SingleStepDynamics(Vector input) {
	return _state;
}


Vector Plant::Evolve(Vector input) {
	_state = SingleStepDynamics(input);
	_input = input;

	return _state;
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