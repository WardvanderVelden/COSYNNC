#pragma once
#include "Plant.h"

namespace COSYNNC {
	class DCDC : public Plant {
	public:
		DCDC() : Plant(2, 1, 0.5, "DCDC", true) { }

		Vector PlantODE(Vector x, float t) override;

		//Vector OverApproximationODE(Vector x, float t) override;
	private:
		float _xc = 70;
		float _xl = 3;
		float _rc = 0.005; // Ohm
		float _rl = 0.05; // Ohm
		float _r0 = 1; // Ohm
		float _vs = 1; // V
	};
}