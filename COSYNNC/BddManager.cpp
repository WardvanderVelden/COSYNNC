#include "BddManager.h"

namespace COSYNNC {
	// Default constructor
	BddManager::BddManager() {

	}


	// Constructor that has pointers to the controller, verifier and quantizers
	BddManager::BddManager(Controller* controller, Verifier* verifier, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		_controller = controller;
		_verifier = verifier;

		_stateQuantizer = stateQuantizer;
		_inputQuantizer = inputQuantizer;

		_stateSpaceWordLength = GetWordLength(_stateQuantizer->GetCardinality());
		_inputSpaceWordLength = GetWordLength(_inputQuantizer->GetCardinality());
	}


	// Save the controller as a BDD for size comparison purposes
	void BddManager::SaveControllerAsBdd(string path, string name) {
		ofstream file(path + "/" + name + ".bdd", std::ios_base::out);

		// Go through all the states and encode the state input pairs in a bdd
		const auto _spaceCardinality = _stateQuantizer->GetCardinality();
		const auto _inputCardinality = _inputQuantizer->GetCardinality();

		// Create BDD and vars
		//Cudd cudd;	

		// Go through all the states and add the representation to the BDD
		for (unsigned long index = 0; index < _spaceCardinality; index++) {
			auto input = _controller->GetInputFromIndex(index);
			auto inputIndex = _inputQuantizer->GetIndexFromVector(input);

			auto stateWord = ComputeBinaryRepresentation(index, _stateSpaceWordLength);
			auto inputWord = ComputeBinaryRepresentation(inputIndex, _inputSpaceWordLength);


		}

		file.close();
	}


	// Compute the binary representation of the index
	vector<bool> BddManager::ComputeBinaryRepresentation(unsigned long index, unsigned int length) {
		vector<bool> word = vector<bool>(length, false);

		for (unsigned int i = (length - 1); i >= 0; i--) {
			auto denominator = pow(2.0, (double)i);

			if ((index - denominator) >= 0) {
				word[i] = true;
				index -= denominator;
			}
		}

		return word;
	}


	// Returns the word length that is required to encode a specific cardinality
	unsigned int BddManager::GetWordLength(unsigned long cardinality) {
		unsigned int wordLength = 1;

		while (pow(2.0, (double)wordLength) < cardinality) {
			wordLength++;
		}

		return wordLength;
	}


	// Test BDDs
	void BddManager::TestBdds() {
		/*Cudd cudd;

		BDD x = cudd.bddVar(0);
		BDD y = cudd.bddVar(1);

		BDD f = x * y;

		std::cout << f << std::endl;*/
	}
}