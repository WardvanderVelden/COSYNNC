#pragma once

#include <iostream>
#include <vector>
#include "Vector.h"
#include "Plant.h"

using namespace std;

namespace COSYNNC {
	class Hyperplane {
	public:
		// Default constructor which generates an empty N-dimensional hyperplane
		Hyperplane(); 

		// Constructor which defines the dimension that the hyperplane exists in
		Hyperplane(unsigned int dimension);

		// Sets the normal and hence defines the normal points of the hyperplane
		void SetNormal(Vector normal, Vector cellCenter);

		// Over approximates the normal of the hyperplane 
		void OverApproximateNormal(Plant* plant, Vector input);

		// Tests if a point in space is on the internal side of the hyperplane
		bool IsPointOnInternalSide(Vector point);

		// Adds a point to the definition of the hyperplane, it is up to the user to ensure it spans a valid plane
		void AddPointToHyperplane(Vector* point);
	private:
		unsigned int _dimension;
		vector<Vector*> _points;

		Vector _normal;
		Vector _normalPoints[2];

		bool _internalSignPositive = true;
	};
}