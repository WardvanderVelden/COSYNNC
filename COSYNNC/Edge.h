#pragma once

#include <iostream>
#include <vector>
#include "Vector.h"

using namespace std;

namespace COSYNNC {
	class Edge {
	public:
		// Default constructor
		Edge();

		// Constructors an edge with the start and end vector as defined here
		Edge(Vector start, Vector end);

		// Transposes the current edge by a delta amount
		void Transpose(Vector delta);

		// Get start vector
		Vector GetStart() const;

		// Get end vector
		Vector GetEnd() const;
	private:
		Vector _start;
		Vector _end;
	};
}