#pragma once

#include <vector>
#include "Vector.h"

using namespace std;

namespace COSYNNC {
	class Transition {
	public:
		// Default constructor
		Transition();

		// Constructor that initializes the transition start
		Transition(long startIndex, unsigned long inputCardinality);

		// Adds an end without checking whether or not the end is already in the transition function
		void AddEnd(long endIndex, unsigned long inputIndex);

		// Returns whether or not an end index is already allocated to that input as a transition
		bool Contains(long endIndex, unsigned long inputIndex);

		// Returns the ends
		vector<long> GetEnds(unsigned long inputIndex) const;

		// Check if the transition that results from that input has already been calculated, if not return false
		bool HasTransitionBeenCalculated(unsigned long inputIndex);
	private:
		long _startIndex = -1;
		vector<long>* _ends = nullptr;

		unsigned long _inputCardinality = 0;
	};
}
