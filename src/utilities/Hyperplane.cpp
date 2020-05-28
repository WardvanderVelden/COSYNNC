#include "Hyperplane.h"

namespace COSYNNC {
	// Default constructor which generates an empty N-dimensional hyperplaned
	Hyperplane::Hyperplane() { 
		_dimension = 0;
	}


	// Constructor which defines the dimension that the hyperplane exists in
	Hyperplane::Hyperplane(unsigned int dimension) {
		_dimension = dimension;
	}


	// Sets the normal and hence defines the normal points of the hyperplane
	void Hyperplane::SetNormal(Vector normal, Vector cellCenter) {
		_normal = normal;
		_normal.Normalize();

		// Calculate center vertex from hyperplane points
		Vector center(_dimension);
		for (unsigned int i = 0; i < _points.size(); i++) {
			center = center + *_points[i];
		}
		center = center * (1 / (float)_points.size());

		// Set normal points
		_normalPoints[0] = center;
		_normalPoints[1] = cellCenter;
	}


	// Over approximates the normal of the hyperplane 
	void Hyperplane::OverApproximateNormal(Plant* plant, Vector input) {
		// Over approximate the normal points
		for (unsigned int i = 0; i < 2; i++) {
			plant->SetState(_normalPoints[i]);
			_normalPoints[i] = plant->EvaluateDynamics(input);
		}

		// Set and normalize the over approximated normal of the plane
		_normal = (_normalPoints[1] - _normalPoints[0]);
		_normal.Normalize();
	}


	// Tests if a point in space is on the internal side of the hyperplane
	bool Hyperplane::IsPointOnInternalSide(Vector point) {
		if (_dimension > 1) {
			// Remove first point of the plane to evaluate difference
			auto diff = point - *_points[0];

			// Evaluate dot product of the projection onto the normal
			auto dotProduct = _normal.Dot(diff);
			//if (!_internalSignPositive) dotProduct *= -1;

			if (dotProduct > 0.0) return true;
		}
		else if (_dimension == 1) {
			if (point[0] > _normal[0] && _internalSignPositive) return true;
			else if (point[0] < _normal[0] && !_internalSignPositive) return true;
		}
		return false;
	}


	// Adds a point to the definition of the hyperplane, it is up to the user to ensure it spans a valid plane
	void Hyperplane::AddPointToHyperplane(Vector* point) {
		_points.push_back(point);
	}
}