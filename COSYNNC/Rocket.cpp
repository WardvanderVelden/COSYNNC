#include "Rocket.h"

namespace COSYNNC {
	// Simple second order rocket dynamics in one axis
	Vector Rocket::StepDynamics(Vector input) {
		Vector newState(GetStateSpaceDimension());

		newState[0] = GetState()[0] + GetState()[1] * GetTau();
		newState[1] = GetState()[1] + GetTau() / _mass * input[0] + _g * GetTau();

		return newState;
	}
}