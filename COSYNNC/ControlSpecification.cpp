#include "ControlSpecification.h";

namespace COSYNNC {
	// Initialize a default control specification
	ControlSpecification::ControlSpecification() {

	}


	// Initializates the control specification based on the control specification type and the plant
	ControlSpecification::ControlSpecification(ControlSpecificationType type, Plant * plant) {
		_type = type;
		_spaceDim = plant->GetStateSpaceDimension();
	}


	// Set the hyper interval for which we want the controller to satisfy the specification
	void ControlSpecification::SetHyperInterval(Vector lowerVertex, Vector upperVertex) {
		_lowerVertex = lowerVertex;
		_upperVertex = upperVertex;

		auto dimension = _lowerVertex.GetLength();
		_center = Vector(dimension);
		for (int i = 0; i < dimension; i++) {
			_center[i] = (_upperVertex[i] - lowerVertex[i]) / 2 + _lowerVertex[i];
		}
	}


	// Checks if the current state vector satisfies is in the specification set
	bool ControlSpecification::IsInSpecificationSet(Vector state) {
		for (int i = 0; i < state.GetLength(); i++) {
			if (state[i] < _lowerVertex[i] || state[i] > _upperVertex[i]) return false;
		}
		return true;
	}


	// Returns the center of the control specification set
	Vector ControlSpecification::GetCenter() const {
		return _center;
	}


	// Returns the control specification type
	ControlSpecificationType ControlSpecification::GetSpecificationType() const {
		return _type;
	}


	// Returns the lower vertex of the control specification set
	Vector ControlSpecification::GetLowerHyperIntervalVertex() const {
		return _lowerVertex;
	}


	// Returns the upper vertex of the control specification set
	Vector ControlSpecification::GetUpperHyperIntervalVertex() const {
		return _upperVertex;
	}


	// Get a random vector from the specified winning space
	Vector ControlSpecification::GetVectorFromSpecification() {
		Vector randomVector(_spaceDim);

		for (int i = 0; i < _spaceDim; i++) {
			float randomFloat = (float)rand() / RAND_MAX;
			randomVector[i] = _lowerVertex[i] + (_upperVertex[i] - _lowerVertex[i]) * randomFloat;
		}

		return randomVector;
	}
}