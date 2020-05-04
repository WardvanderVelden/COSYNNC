#include "StateSpaceRepresentation.h"

namespace COSYNNC {
	// Default deconstructor
	StateSpaceRepresentation::~StateSpaceRepresentation() {
		for (size_t i = 0; i < _stateSpaceDimension; i++) delete[] _A[i];
		for (size_t i = 0; i < _inputSpaceDimension; i++) delete[] _B[i];

		delete[] _A;
		delete[] _B;
	}

	// Sets the A and B matrices of the plant which define the dynamics
	void StateSpaceRepresentation::SetMatrices(double** A, double** B) {
		_A = new double*[_stateSpaceDimension];
		for (size_t j = 0; j < _stateSpaceDimension; j++) {
			_A[j] = new double[_stateSpaceDimension];
			for (size_t i = 0; i < _stateSpaceDimension; i++) _A[j][i] = A[j][i];
		}

		_B = new double* [_stateSpaceDimension];
		for (size_t j = 0; j < _stateSpaceDimension; j++) {
			_B[j] = new double[_inputSpaceDimension];
			for (size_t i = 0; i < _inputSpaceDimension; i++) _B[j][i] = B[j][i];
		}
	}


	// The ordinary differential equation which is fed to the numerical integrator and which define the plant dynamics
	Vector StateSpaceRepresentation::DynamicsODE(Vector x, float t) {
		Vector dxdt = Vector(_stateSpaceDimension);

		// A*x
		for (size_t j = 0; j < _stateSpaceDimension; j++) {
			for (size_t i = 0; i < _stateSpaceDimension; i++) {
				auto value = _A[j][i];
				dxdt[j] += _A[j][i] * x[i];
			}
		}

		// B*u
		for (size_t j = 0; j < _stateSpaceDimension; j++) {
			for (size_t i = 0; i < _inputSpaceDimension; i++) {
				dxdt[j] += _B[j][i] * _u[i];
			}
		}

		return dxdt;
	}
}