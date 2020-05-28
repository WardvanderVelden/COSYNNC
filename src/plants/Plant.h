#pragma once
#include <functional>
#include "Vector.h"

using namespace COSYNNC;

namespace COSYNNC {
	class Plant
	{
	public:
		// Initializes the plant based on the state space dimension, the input space dimension and the time step
		Plant(unsigned int stateSpaceDimension = 0, unsigned int inputSpaceDimension = 0, double tau = 0.1, string name = "Plant", bool isLinear = false);

		// Defines the dynamics of the plant for a single time step tau
		virtual Vector EvaluateDynamics(Vector input);

		// Defines the over approximation of the dynamics of the plant for a single time step
		virtual Vector EvaluateRadialGrowthBound(Vector r, Vector input);

		// Evolves the plant based on the single step dynamics and the input based on time step
		void Evolve(Vector input);

		// Returns the current state of the plant
		Vector GetState() const;

		// Sets the state of the plant in the state space
		void SetState(Vector newState);

		// Returns the state space dimension
		int GetStateSpaceDimension() const { return _stateSpaceDimension; }

		// Returns the input space dimension
		int GetInputSpaceDimension() const { return _inputSpaceDimension; }

		// Returns the time step size
		float GetStepSize() const { return _h; }

		// Returns if the plant is linear
		bool IsLinear() const { return _isLinear; }

		// Returns the name of the plant
		string GetName() const { return _name; }

		// Prints the current state of the system
		void PrintState() const;

		// Solve the ODE using fourth order Runge Kutta
		Vector SolveRK4(Vector x, float t, unsigned int steps, bool overApproximation = false);

	protected:
		const unsigned int _stateSpaceDimension;
		const unsigned int _inputSpaceDimension;

		const bool _isLinear;

		const string _name;

		const double _h;

		Vector _x;
		Vector _u;

	private:
		// Ordinary differential equation that describes the plant
		virtual Vector DynamicsODE(Vector x, float t);

		// Ordinary differential equation that describes the plant over approximation of the plant
		virtual Vector RadialGrowthBoundODE(Vector r, float t);
	};
}


