#include "Plant.h"

using namespace COSYNNC;

Plant::Plant(unsigned int stateSpaceDimension, unsigned int inputSpaceDimension, double h, string name, bool isLinear) :
	_stateSpaceDimension(stateSpaceDimension), 
	_inputSpaceDimension(inputSpaceDimension),
	_h(h), 
	_name(name), 
	_isLinear(isLinear) {

	_x.SetLength(stateSpaceDimension);
	_u.SetLength(inputSpaceDimension);
}


// Virtual function that describes the plant dynamics subject to a time step of tau, should be overriden by the actual plant dynamics
Vector Plant::EvaluateDynamics(Vector input) {
	// Set the input to the specified input
	_u = input;
	
	// Return the runge kutta integrated new state
	return SolveRK4(_x, 0.0, 4, false);
}


// Defines the over approximation of the dynamics of the plant for a single time step tau
Vector Plant::EvaluateRadialGrowthBound(Vector r, Vector input) {
	// Set the input to the specified input
	_u = input;

	// Return the runge kutta integrated over approximation
	return SolveRK4(r, 0.0, 4, true);
}


// Evolves the plant based on the single step dynamics and the input based on time step
void Plant::Evolve(Vector input) {
	_x = EvaluateDynamics(input);
}


Vector Plant::GetState() const {
	return _x;
}


void Plant::SetState(Vector state) {
	_x = state;
}


void Plant::PrintState() const {
	std::cout << "Plant state:" << std::endl;
	_x.PrintValues();
}


// Solve the ODE using fourth order Runge Kutta
Vector Plant::SolveRK4(Vector x, float t, unsigned int steps, bool overApproximation) {
	float ch = _h / steps;

	for (unsigned int i = 0; i < steps; i++) {
		// Get all the k factors
		Vector k1 = (overApproximation) ? RadialGrowthBoundODE(x,t) * ch : DynamicsODE(x, t) * ch;

		auto k1half = k1 / 2;
		Vector k2 = (overApproximation) ? RadialGrowthBoundODE(x + k1half, t + ch / 2) * ch : DynamicsODE(x + k1half, t + ch / 2) * ch;

		auto k2half = k2 / 2;
		Vector k3 = (overApproximation) ? RadialGrowthBoundODE(x + k2half, t + ch / 2) * ch : DynamicsODE(x + k2half, t + ch / 2) * ch;

		Vector k4 = (overApproximation) ? RadialGrowthBoundODE(x + k3, t + ch) * ch : DynamicsODE(x + k3, t + ch) * ch;

		// Average out to find the approximation of the integration
		auto k22 = k2 * 2;
		auto k32 = k3 * 2;

		auto average = (k1 + k22 + k32 + k4);
		average = average * (1.0 / 6.0);

		// Update x and t
		x = x + average;
		t = t + ch;
	}

	// Return the last x
	return x;
}


// Ordinary differential equation that describes the plant
Vector Plant::DynamicsODE(Vector x, float t) {
	// Default is just ones (no change)
	Vector ones(_stateSpaceDimension);
	for (unsigned int i = 0; i < _stateSpaceDimension; i++) ones[i] = 1.0;

	return ones;
}


// Ordinary differential equation that describes the plant over approximation of the plant
Vector Plant::RadialGrowthBoundODE(Vector x, float t) {
	// Default over approximation is simply that of the plant ODE.
	return DynamicsODE(x, t); 
}