#pragma once
#include "Plant.h"

namespace COSYNNC {
	class LinearHybrid : public Plant {
	public:
		LinearHybrid() : Plant(2, 1, 0.2, "Linear Hybrid", true) { }

		Vector DynamicsODE(Vector x, float t) override;

		//Vector RadialGrowthBoundODE(Vector x, float t) override;
	};
}