#pragma once
#include "Vector.h"
#include "Plant.h"

namespace COSYNNC {
	enum class ControlSpecificationType {
		Invariance,
		Reachability
	};

	class ControlSpecification {
	public:
		// Initialize a default control specification
		ControlSpecification();

		// Initializates the control specification based on the control specification type and the plant
		ControlSpecification(ControlSpecificationType type, Plant * plant);

		// Set the hyper interval for which the controller should satisfy the specification
		void SetHyperInterval(Vector lowerVertex, Vector upperVertex);

		// Gets the center of the hyper interval
		Vector GetCenter() const;
	private:
		ControlSpecificationType _type;

		int _spaceDim;

		Vector _lowerVertex;
		Vector _upperVertex;

		// DEBUG: Temporary to make a simple control
		Vector _center;
	};
}