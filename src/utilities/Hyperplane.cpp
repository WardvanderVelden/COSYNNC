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


	// Computes the normal based on the hyperplane
	void Hyperplane::ComputeNormal(Vector center) {
		if (_dimension == 2) {
			Vector dir = *_points[1] - *_points[0];

			_normal[0] = dir[1];
			_normal[1] = -dir[0];
		} 
		else if (_dimension == 3) {
			Vector one = *_points[1] - *_points[0];
			Vector two = *_points[2] - *_points[0];

			one.Normalize();
			two.Normalize();

			_normal[0] = one[1] * two[2] - two[1] * one[2];
			_normal[1] = one[0] * two[2] - two[0] * one[2];
			_normal[2] = one[0] * two[1] - two[0] * one[1];
		}
		else {
			// TODO: Add higher dimensions
		}

		// Normalize
		_normal.Normalize();

		// Make sure the normal is directed at the center
		if (!IsPointOnInternalSide(center)) _normal = _normal * -1;
	}


	// Tests if a point in space is on the internal side of the hyperplane
	bool Hyperplane::IsPointOnInternalSide(Vector point) {
		if (_dimension > 1) {
			// Remove first point of the plane to evaluate difference
			auto diff = point - *_points[0];

			// Evaluate dot product of the projection onto the normal
			auto dotProduct = _normal.Dot(diff);

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