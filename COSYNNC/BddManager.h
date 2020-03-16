#pragma once

#include <fstream>
#include <vector>
#include <string>

#include "StringHelper.h"
#include "NeuralNetwork.h"
#include "Verifier.h"

#include "cuddObj.hh"

using namespace std;

namespace COSYNNC {
	class BddManager {
	public:
		// Default constructor
		BddManager();

		// Constructor that has pointers to the controller, verifier and quantizers
		BddManager(Controller* controller, Verifier* verifier, Quantizer* stateQuantizer, Quantizer* inputQuantizer);

		// Save the controller as a BDD for size comparison purposes
		void SaveControllerAsBdd(string path, string name);

		// Compute the binary representation of the index
		vector<bool> ComputeBinaryRepresentation(unsigned long index, unsigned int length);

		// Returns the word length that is required to encode a specific cardinality
		unsigned int GetWordLength(unsigned long cardinality);

		// Test BDDs
		void TestBdds();
	private:
		Quantizer* _stateQuantizer = nullptr;
		Quantizer* _inputQuantizer = nullptr;

		Verifier* _verifier = nullptr;

		Controller* _controller = nullptr;

		unsigned int _stateSpaceWordLength = 0;
		unsigned int _inputSpaceWordLength = 0;
	};
}
