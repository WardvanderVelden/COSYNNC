#pragma once

#include <iostream>
#include <vector>
#include "Vector.h"

using namespace std;

namespace COSYNNC {
	class Plane {
	public:
		// Default constructor which generates an empty N-dimensional plane
		Plane(); 

		// Constructor which defines the plane through a set of points
		Plane(vector<Vector> points);

		// Constructor which defines the plane through a set of points and define the internal side
		Plane(vector<Vector> points, Vector internalPoint);

		// Calculates the normal to the plane
		void CalculateNormal();

		// Defines the internal side of the plane by assigning a point in space which will then become the internal side
		void SetInternalSide(Vector point);

		// Tests if a point in space is on the internal side of the plane
		bool IsPointOnInternalSide(Vector point);
	private:
		unsigned int _dimension;
		vector<Vector> _points;

		Vector _normal;

		bool _internalSignPositive = true;
	};
}