#include "DCDC.h"

namespace COSYNNC {
	Vector DCDC::StepDynamics(Vector input) {
		auto stateDim = GetStateSpaceDimension();
		auto tau = GetTau();

		Vector derivative(stateDim);
		auto state = GetState();

		if (input[0] < 0.5) {
			derivative[0] = (-_rl / _xl) * state[0] + _vs / _xl;
			derivative[1] = ((-1 / _xc) * (1 / (_r0 + _rc))) * state[1];
		}
		else {
			derivative[0] = ((-1 / _xl) * (_rl + ((_r0 * _rc) / (_r0 + _rc)))) * state[0] + (((-1 / _xl) * (_r0 / (_r0 + _rc))) / 5) * state[1] + _vs / _xl;
			derivative[1] = (5 * (_r0 / (_r0 + _rc)) * (1 / _xc)) * state[0] + ((-1 / _xc) * (1 / (_r0 + _rc))) * state[1];
		}

		Vector newState(stateDim);
		newState = state + derivative * tau;

		return newState;
	}
}