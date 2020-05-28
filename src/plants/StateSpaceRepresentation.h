#pragma once
#include "Plant.h"

namespace COSYNNC {
	class StateSpaceRepresentation : public Plant {
	public:
		StateSpaceRepresentation(unsigned int stateSpaceDimension, unsigned int inputSpaceDimension, double h, string name) : Plant(stateSpaceDimension, inputSpaceDimension, h, name, true) { };

		// Default deconstructor
		~StateSpaceRepresentation();

		// Sets the A and B matrices of the plant which define the dynamics
		void SetMatrices(double** A, double** B);

		// The ordinary differential equation which is fed to the numerical integrator and which define the plant dynamics
		Vector DynamicsODE(Vector x, float t) override;
	private:
		double** _A = nullptr;
		double** _B = nullptr;
	};
}
