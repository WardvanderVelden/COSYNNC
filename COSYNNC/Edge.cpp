#include "Edge.h"

namespace COSYNNC {
	// Default constructor
	Edge::Edge() { }


	// Constructors an edge with the start and end vector as defined here
	Edge::Edge(Vector start, Vector end) {
		_start = start;
		_end = end;
	}


	// Transposes the current edge by a delta amount
	void Edge::Transpose(Vector delta) {
		auto length = _start.GetLength();

		for (unsigned int i = 0; i < length; i++) {
			_start[i] += delta[i];
			_end[i] += delta[i];
		}
	}


	// Get start vector
	Vector Edge::GetStart() const {
		return _start;
	}


	// Get end vector
	Vector Edge::GetEnd() const {
		return _end;
	}
}