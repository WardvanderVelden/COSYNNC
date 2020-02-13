#include "DCDC.h"

namespace COSYNNC {
	Vector DCDC::PlantODE(Vector x, float t) {
		Vector dxdt = Vector(_stateSpaceDim);

		if (_u[0] < 0.5) {
			dxdt[0] = (-_rl / _xl) * x[0];
			dxdt[1] = ((-1 / _xc) * (1 / (_r0 + _rc))) * x[1];
		}
		else {
			dxdt[0] = ((-1 / _xl) * (_rl + ((_r0 * _rc) / (_r0 + _rc)))) * x[0] + (((-1 / _xl) * (_r0 / (_r0 + _rc))) / 5) * x[1];
			dxdt[1] = ((5 * (_r0 / (_r0 + _rc)) * (1 / _xc))) * x[0] + ((-1 / _xc) * (1 / (_r0 + _rc))) * x[1];
		}

		// Add bias
		dxdt[0] = dxdt[0] + _vs / _xl;

		return dxdt;
	}


	/*Vector DCDC::OverApproximationODE(Vector x, float t) {
		Vector dxdt = Vector(_stateSpaceDim);

		if (_u[0] < 0.5) {
			dxdt[0] = (-_rl / _xl) * x[0];
			dxdt[1] = ((-1 / _xc) * (1 / (_r0 + _rc))) * x[1];
		}
		else {
			//dxdt[0] = ((-1 / _xl) * (_rl + ((_r0 * _rc) / (_r0 + _rc)))) * x[0] + (((-1 / _xl) * (_r0 / (_r0 + _rc))) / 5) * x[1];
			//dxdt[1] = (5 * (_r0 / (_r0 + _rc)) * (1 / _xc)) * x[0] + ((-1 / _xc) * (1 / (_r0 + _rc))) * x[1];
			dxdt[0] = ((-1 / _xl) * (_rl + ((_r0 * _rc) / (_r0 + _rc)))) * x[0] + (((1 / _xl) * (_r0 / (_r0 + _rc))) / 5) * x[1];
			dxdt[1] = (5 * (_r0 / (_r0 + _rc)) * (1 / _xc)) * x[0] + ((-1 / _xc) * (1 / (_r0 + _rc))) * x[1];
		}

		return dxdt;
	}*/
}