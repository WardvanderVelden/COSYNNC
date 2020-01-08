#pragma once

#include <vector>

using namespace std;

namespace COSYNNC {
	class Transition {
	public:
		// Default constructor
		Transition();

		// Constructor that initializes the transition start
		Transition(long start);

		// Returns true if the end is already contained in this transition
		bool Contains(long end);

		// Adds an end
		void AddEnd(long end);

		// Returns the amount of ends
		unsigned int GetAmountOfEnds() const;

		// Returns the ends
		vector<long> GetEnds() const;
	private:
		long _start = -1;

		vector<long> _ends;
		unsigned int _amountOfEnds = 0;
	};
}
