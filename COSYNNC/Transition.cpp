#include "Transition.h"

namespace COSYNNC {
	// Default constructor
	Transition::Transition() { }


	// Constructor that initializes the transition start
	Transition::Transition(long start) {
		_start = start;
	}


	// Returns true if the end is already contained in this transition
	bool Transition::Contains(long end) {
		for (auto index : _ends) {
			if (index == end) return true;
		}
		return false;
	}


	// Adds an end
	void Transition::AddEnd(long end) {
		if (!Contains(end)) {
			_ends.push_back(end);
			_amountOfEnds++;
		}
	}


	// Returns the amount of ends
	unsigned int Transition::GetAmountOfEnds() const {
		return _amountOfEnds;
	}


	// Returns the ends
	vector<long> Transition::GetEnds() const {
		return _ends;
	}
}