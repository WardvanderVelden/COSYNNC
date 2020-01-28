#include "Rocket.h"

namespace COSYNNC {
	// Simple second order rocket dynamics in one axis
	Vector Rocket::EvaluateDynamics(Vector input) {
		_u = input;

		Vector newState(_stateSpaceDim);

		newState[0] = _x[0] + _x[1] * _h;
		newState[1] = _x[1] + _h / _mass * _u[0] + _g * _h;

		return newState;
	}
}