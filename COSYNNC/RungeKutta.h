#pragma once

#include <iostream>
#include <vector>
#include "Vector.h"

using namespace std;

namespace COSYNNC {
	class RungeKutta {
	public:
		// Default constructor
		RungeKutta();

		// Solve a differential equation using four order Runge Kutta
		virtual Vector SolveRK4();
	};
}