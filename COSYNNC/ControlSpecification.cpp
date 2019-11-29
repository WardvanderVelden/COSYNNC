#include "ControlSpecification.h";

namespace COSYNNC {
	ControlSpecification::ControlSpecification() {

	}


	ControlSpecification::ControlSpecification(ControlSpecificationType type, Plant * plant) {
		_type = type;
		_spaceDim = plant->GetStateSpaceDimension();
	}


	// Set the hyper interval for which we want the controller to satisfy the specification
	void ControlSpecification::SetHyperInterval(Vector lowerVertex, Vector upperVertex) {
		_lowerVertex = lowerVertex;
		_upperVertex = upperVertex;
	}


	// Checks if the current state vector satisfies is in the control specification goal
	bool ControlSpecification::IsInControlGoal(Vector state) {
		for (int i = 0; i < state.GetLength(); i++) {
			if (state[i] < _lowerVertex[i] || state[i] > _upperVertex[i]) return false;
		}
		return true;
	}
}