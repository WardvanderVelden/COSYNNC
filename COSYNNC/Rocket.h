#pragma once
#include "Plant.h"

namespace COSYNNC {
	class Rocket : public Plant {
	public:
		Rocket() : Plant(2, 1, 0.1) { }

		// Simple second order rocket dynamics in one axis
		Vector SingleStepDynamics(Vector input) override;
	private:
		const float _mass = 267; // kg
		const float _g = -9.81; // m s^-2
	};
}