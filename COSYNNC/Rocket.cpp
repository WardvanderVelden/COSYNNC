#include "Rocket.h"

namespace COSYNNC {
	// Simple second order rocket dynamics in one axis
	Vector Rocket::PlantODE(Vector x, float t) {
		Vector dxdt = Vector(_stateSpaceDim);

		dxdt[0] = x[1];
		dxdt[1] = _u[0]/_mass + _g;

		return dxdt;
	}
}