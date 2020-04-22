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

		// Check if the transition that results from that input has already been calculated, if not return false
		bool HasTransitionBeenCalculated(unsigned long inputIndex);

		// Returns the ends for a specific input
		vector<long> GetEnds(unsigned long inputIndex) const;

		// Returns the processed inputs
		vector<unsigned long> GetProcessedInputs() const;

		// Returns the post for a certain input
		Vector GetPost(unsigned long inputIndex) const;

		// Returns the lower bound for a certain input
		Vector GetLowerBound(unsigned long inputIndex) const;

		// Returns the upper bound for a certain input
		Vector GetUpperBound(unsigned long inputIndex) const;

		// Sets the input to have been processed for that transition
		void SetInputProcessed(unsigned long inputIndex);

		// Sets the absolute post of a transition
		void SetPost(Vector post, unsigned long inputIndex);

		// Sets the lower and upper bound for a given input 
		void SetLowerAndUpperBound(Vector lower, Vector upper, unsigned long inputIndex);
	private:
		long _startIndex = -1;
		vector<long>* _ends = nullptr;

		vector<unsigned long> _processedInputs;

		unsigned long _inputCardinality = 0;

		// Lower and upper bounds of the transitions for every input (for SCOTS)
		vector<Vector> _post;
		vector<Vector> _lowerBounds;
		vector<Vector> _upperBounds;
	};
}
