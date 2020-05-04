#pragma once
#include "Vector.h"
#include "Plant.h"

namespace COSYNNC {
	enum class ControlSpecificationType {
		Invariance,
		Reachability,
		ReachAndStay
	};

	class ControlSpecification {
	public:
		// Initialize a default control specification
		ControlSpecification();

		// Initializates the control specification based on the control specification type and the plant
		ControlSpecification(ControlSpecificationType type, Plant * plant);

		// Set the hyper interval for which the controller should satisfy the specification
		void SetHyperInterval(Vector lowerVertex, Vector upperVertex);

		// Checks if the current state vector satisfies is in the control specification goal
		bool IsInSpecificationSet(Vector state); 

		// Returns the center of the control specification set
		Vector GetCenter() const;

		// Returns the control specification type
		ControlSpecificationType GetSpecificationType() const;

		// Returns the lower vertex of the control specification set
		Vector GetLowerHyperIntervalVertex() const;

		// Returns the upper vertex of the control specification set
		Vector GetUpperHyperIntervalVertex() const;

		// Get a random vector from the specified winning space
		Vector GetVectorFromSpecification();
	private:
		ControlSpecificationType _type;

		int _spaceDimension;

		Vector _lowerVertex;
		Vector _upperVertex;

		Vector _center;
	};
}