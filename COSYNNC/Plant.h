#pragma once
#include <functional>
#include "Vector.h";

using namespace COSYNNC;

namespace COSYNNC {
	class Plant
	{
	public:
		Plant(int stateSpaceDimension = 0, int inputSpaceDimension = 0, float tau = 0.1);

		virtual Vector SingleStepDynamics(Vector input);

		void Evolve(Vector input);

		Vector GetState() const;
		void SetState(Vector newState);

		int GetStateSpaceDimension() const { return _stateSpaceDim; }
		int GetInputSpaceDimension() const { return _inputSpaceDim; }

		float GetTau() const { return _tau; }

		void PrintState() const;
		
	private:
		const int _stateSpaceDim;
		const int _inputSpaceDim;

		const float _tau;

		Vector _state;
		Vector _input; // input that led to current state
	};
}


