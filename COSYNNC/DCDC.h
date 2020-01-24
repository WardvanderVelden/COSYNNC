#pragma once
#include "Plant.h"

namespace COSYNNC {
	class DCDC : public Plant {
	public:
		//DCDC() : Plant(2, 1, 0.25) { }
		DCDC() : Plant(2, 1, 0.1) { }

		Vector StepDynamics(Vector input) override; // Dynamics are described by two modes, hence input is binary ( < 0.5, > 0.5)

		Vector StepOverApproximation(Vector input) override;
	private:
		float _xc = 70;
		float _xl = 3;
		float _rc = 0.005; // Ohm
		float _rl = 0.05; // Ohm
		float _r0 = 1; // Ohm
		float _vs = 1; // V
	};
}