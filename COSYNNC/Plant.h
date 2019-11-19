#pragma once
#include <functional>
#include "Vector.h";

using namespace cosynnc;

namespace cosynnc {
	class Plant
	{
	public:
		Plant(int stateDimension = 0, int inputDimension = 0, float tau = 0.1);

		virtual Vector SingleStepDynamics(Vector input);

		Vector Evolve(Vector input);

		Vector GetState() const;
		void SetState(Vector newState);

		int GetStateDimension() const { return _stateDim; }
		int GetInputDimension() const { return _inputDim; }

		float GetTau() const { return _tau; }

		void PrintState() const;
		
	private:
		const int _stateDim;
		const int _inputDim;

		const float _tau;

		Vector _state;
		Vector _input; // input that led to current state
	};
}


