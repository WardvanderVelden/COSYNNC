#pragma once
#include "Plant.h"

#define PI 3.14159265

namespace COSYNNC {
	class Unicycle : public Plant {
	public:
		Unicycle() : Plant(3, 1, 0.3, "Unicycle", false) { };

		Vector EvaluateDynamics(Vector input) override;

		Vector DynamicsODE(Vector x, float t) override;

		Vector RadialGrowthBoundODE(Vector r, float t) override;
	private:
		double _v = 1.0;
		double _omega = 2.0;
	};
}
