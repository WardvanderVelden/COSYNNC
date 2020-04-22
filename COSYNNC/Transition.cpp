#include "Transition.h"

namespace COSYNNC {
	// Default constructor
	Transition::Transition() { }


	// Constructor that initializes the transition start
	Transition::Transition(long startIndex, unsigned long inputCardinality) {
		_startIndex = startIndex;
		_ends = new vector<long>[inputCardinality];

		_post = vector<Vector>(inputCardinality);
		_lowerBounds = vector<Vector>(inputCardinality);
		_upperBounds = vector<Vector>(inputCardinality);

		for (unsigned long index = 0; index < inputCardinality; index++) {
			_ends[index] = vector<long>();

			_post[index] = Vector();
			_lowerBounds[index] = Vector();
			_upperBounds[index] = Vector();
		}
	}


	// Adds an end without checking whether or not the end is already in the transition function
	void Transition::AddEnd(long endIndex, unsigned long inputIndex) {
		// TODO: Checking if it is contained already is relatively expensive and can probably be done in a more efficient manner
		if (!Contains(endIndex, inputIndex)) { 
			_ends[inputIndex].push_back(endIndex);
		}
	}


	// Returns whether or not an end index is already allocated to that input as a transition
	bool Transition::Contains(long endIndex, unsigned long inputIndex) {
		auto endIndices = _ends[inputIndex];
		for (auto end : endIndices) {
			if (end == endIndex) return true;
		}
		return false;
	}


	// Check if the transition that results from that input has already been calculated, if not return false
	bool Transition::HasTransitionBeenCalculated(unsigned long inputIndex) {
		if (_ends[inputIndex].size() > 0) return true;

		return false;
	}


	// Returns the ends
	vector<long> Transition::GetEnds(unsigned long inputIndex) const {
		if (_ends[inputIndex].size() > 1) return _ends[inputIndex];
		else if (_ends[inputIndex][0] != -1) return _ends[inputIndex];

		return vector<long>();
	}


	// Returns the processed inputs
	vector<unsigned long> Transition::GetProcessedInputs() const {
		return _processedInputs;
	}


	// Returns the post for a certain input
	Vector Transition::GetPost(unsigned long inputIndex) const {
		return _post[inputIndex];
	}


	// Returns the lower bound for a certain input
	Vector Transition::GetLowerBound(unsigned long inputIndex) const {
		return _lowerBounds[inputIndex];
	}


	// Returns the upper bound for a certain input
	Vector Transition::GetUpperBound(unsigned long inputIndex) const {
		return _upperBounds[inputIndex];
	}


	// Sets the input to have been processed for that transition
	void Transition::SetInputProcessed(unsigned long inputIndex) {
		_processedInputs.push_back(inputIndex);
	}

	// Sets the absolute post of a transition
	void Transition::SetPost(Vector post, unsigned long inputIndex) {
		_post[inputIndex] = post;
	}
	

	// Sets the lower and upper bound for a given input 
	void Transition::SetLowerAndUpperBound(Vector lower, Vector upper, unsigned long inputIndex) {
		_lowerBounds[inputIndex] = lower;
		_upperBounds[inputIndex] = upper;
	}
}