#pragma once
#include "Plant.h"

namespace COSYNNC {
	class Rocket : public Plant {
	public:
		Rocket() : Plant(2, 1, 0.1, "Rocket", true) { }

		Vector DynamicsODE(Vector x, float t) override;
	private:
		const float _mass = 267; // kg
		const float _g = -9.81; // m s^-2
	};
}