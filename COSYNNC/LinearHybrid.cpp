#include "LinearHybrid.h"

namespace COSYNNC {
	Vector LinearHybrid::DynamicsODE(Vector x, float t) {
		Vector dxdt = Vector(_stateSpaceDimension);

		// Linear hybrid dynamics
		if (_u[0] < 1.0) {
			dxdt[0] = 1 * x[0] +  0 * x[1];
			dxdt[1] = 0 * x[0] + -2.5 * x[1];
		}
		else {
			dxdt[0] = -2.5 * x[0] + 0 * x[1];
			dxdt[1] = 0 * x[0] + 1 * x[1];
		}

		return dxdt;
	}
}