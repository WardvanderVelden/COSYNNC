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

		_center = _lowerVertex + (_upperVertex - _lowerVertex) * 0.5;
	}

	// Gets the center of the hyper interval
	Vector ControlSpecification::GetCenter() const {
		return _center;
	}
}