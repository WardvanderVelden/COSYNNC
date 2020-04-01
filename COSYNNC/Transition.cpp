#include "Transition.h"

namespace COSYNNC {
	// Default constructor
	Transition::Transition() { }


	// Constructor that initializes the transition start
	Transition::Transition(long startIndex, unsigned long inputCardinality) {
		_startIndex = startIndex;
		_ends = new vector<long>[inputCardinality];

		for (unsigned long index = 0; index < inputCardinality; index++) {
			_ends[index] = vector<long>();
		}
	}


	// Adds an end without checking whether or not the end is already in the transition function
	void Transition::AddEnd(long endIndex, unsigned long inputIndex) {
		auto ends = _ends[inputIndex];
		_ends[inputIndex].push_back(endIndex);
	}


	// Returns whether or not an end index is already allocated to that input as a transition
	bool Transition::Contains(long endIndex, unsigned long inputIndex) {
		auto endIndices = _ends[inputIndex];
		for (auto end : endIndices) {
			if (end == endIndex) return true;
		}
		return false;
	}


	// Returns the ends
	vector<long> Transition::GetEnds(unsigned long inputIndex) const {
		return _ends[inputIndex];
	}


	// Check if the transition that results from that input has already been calculated, if not return false
	bool Transition::HasTransitionBeenCalculated(unsigned long inputIndex) {
		if (_ends[inputIndex].size() > 0) return true;

		return false;
	}
}