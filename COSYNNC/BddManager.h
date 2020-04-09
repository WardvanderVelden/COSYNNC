#pragma once

#include <fstream>
#include <vector>
#include <string>

//#include "cuddObj.hh"
//#include "cudd.h"
//#include "dddmp.h"


#include "StringHelper.h"
#include "NeuralNetwork.h"
#include "Verifier.h"

using namespace std;

namespace COSYNNC {
	class BddManager {
	public:
		// Default constructor
		BddManager();

		// Constructor that has pointers to the controller, verifier and quantizers
		BddManager(Abstraction* abstraction);

		// Save the controller as a BDD for size comparison purposes
		void SaveControllerAsBdd(string path, string name);

		// Test BDDs
		void TestBdds();
	private:
		// Compute the binary representation of the index
		vector<bool> ComputeBinaryRepresentation(unsigned long index, unsigned int length);

		// Returns the word length that is required to encode a specific cardinality
		unsigned int GetWordLength(unsigned long cardinality);


		Abstraction* _abstraction = nullptr;

		unsigned int _stateSpaceWordLength = 0;
		unsigned int _inputSpaceWordLength = 0;
	};
}
