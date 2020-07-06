#include "Unicycle.h"
//#include "Quantizer.h"

namespace COSYNNC {
	Vector Unicycle::EvaluateDynamics(Vector input) {
		// Set the input to the specified input
		_u = input;

		// Return the runge kutta integrated new state
		auto newState = SolveRK4(_x, 0.0, 4, false);

		// Modulate angle back into range
		if (newState[2] >= (2 * PI)) newState[2] -= (2 * PI);
		if (newState[2] < 0) newState[2] += (2 * PI);

		return newState;
	}


	Vector Unicycle::DynamicsODE(Vector x, float t) {
		Vector dxdt(_stateSpaceDimension);

		dxdt[0] = _v * cos(x[2]);
		dxdt[1] = _v * sin(x[2]);
		dxdt[2] = _u[0] * _omega;

		return dxdt;
	}


	Vector Unicycle::RadialGrowthBoundODE(Vector r, float t) {
		Vector drdt(_stateSpaceDimension);

		drdt[0] = r[2] * _v;
		drdt[1] = r[2] * _v;
		drdt[2] = 0;

		return drdt;
	}
}