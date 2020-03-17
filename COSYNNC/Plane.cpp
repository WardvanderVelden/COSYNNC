#include "Plane.h"

namespace COSYNNC {
	// Default constructor which generates an empty N-dimensional plane
	Plane::Plane() { 
		_dimension = 0;
	}


	// Constructor which defines the plane through a set of points
	Plane::Plane(vector<Vector> points) {
		_dimension = points.size();
		_points = points;

		CalculateNormal();
	}


	// Constructor which defines the plane through a set of points and define the internal side
	Plane::Plane(vector<Vector> points, Vector internalPoint) {
		_dimension = points.size();
		_points = points;

		CalculateNormal();
		SetInternalSide(internalPoint);
	}


	// Calculates the normal to the plane
	void Plane::CalculateNormal() {
		// TODO: Turn this into a generic N-dimensional method of calculating the normal instead of case wise methods
		switch (_dimension) {
			case 0: {
				_normal = Vector(0);
				break;
				}
			case 1: {
				_normal = _points[0];
				break;
			}
			case 2: {
				auto diff = _points[1] - _points[0];
				_normal = Vector({ -diff[1], diff[0] });
				break;
			}
			case 3: {
				auto vec1 = _points[1] - _points[0];
				auto vec2 = _points[2] - _points[1];

				_normal = Vector({vec1[1] * vec2[2] - vec1[2] * vec2[1], vec1[2] * vec2[0] - vec1[0] * vec2[2], vec1[0] * vec2[1] - vec1[1] * vec2[0]}); // Cross product in 3d
				break;
			}
		}

		_normal.Normalize();
	}


	// Defines the internal side of the plane by assigning a point in space which will then become the internal side
	void Plane::SetInternalSide(Vector point) {
		if (!IsPointOnInternalSide(point)) _internalSignPositive = false;
	}


	// Tests if a point in space is on the internal side of the plane
	bool Plane::IsPointOnInternalSide(Vector point) {
		if (_dimension > 1) {
			// Remove first point of the plane to evaluate difference
			auto diff = point - _points[0];

			// Evaluate dot product of the projection onto the normal
			auto dotProduct = _normal.Dot(diff);
			if (!_internalSignPositive) dotProduct *= -1;

			if (dotProduct >= 0.0) return true;
		}
		else if (_dimension == 1) {
			if (point[0] >= _normal[0] && _internalSignPositive) return true;
			else if (point[0] <= _normal[0] && !_internalSignPositive) return true;
		}
		return false;
	}
}