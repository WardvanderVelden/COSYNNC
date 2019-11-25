#pragma once
#include <functional>
#include "Vector.h";

using namespace COSYNNC;

namespace COSYNNC {
	class Plant
	{
	public:
		// Initializes the plant based on the state space dimension, the input space dimension and the time step tau
		Plant(int stateSpaceDimension = 0, int inputSpaceDimension = 0, float tau = 0.1);


		// Defines the dynamics of the plant for a single time step tau
		virtual Vector SingleStepDynamics(Vector input);

		// Evolves the plant based on the single step dynamics and the input based on time step tau
		void Evolve(Vector input);

		// Returns the current state of the plant
		Vector GetState() const;

		// Sets the state of the plant in the state space
		void SetState(Vector newState);

		// Returns the state space dimension
		int GetStateSpaceDimension() const { return _stateSpaceDim; }

		// Returns the input space dimension
		int GetInputSpaceDimension() const { return _inputSpaceDim; }

		// Returns the time step tau
		float GetTau() const { return _tau; }

		// Prints the current state of the system
		void PrintState() const;
		
	private:
		const int _stateSpaceDim;
		const int _inputSpaceDim;

		const float _tau;

		Vector _state;
		Vector _input; // input that led to current state
	};
}


