#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <math.h>

#include "dddmp.h"
#include "cuddObj.hh"

#define ULONG_MAX  4294967295UL


// Cardinality of the state and input space
const unsigned long stateCardinality = 10000;
const unsigned long inputCardinality = 10;

// Constants
const unsigned int stateWordLength = ceil(log(stateCardinality) / log(2));
const unsigned int inputWordLength = ceil(log(inputCardinality) / log(2));

const bool verboseMode = false;


// Reads a static controller and converts it into a BDD
unsigned long readStaticController(unsigned long* controller, std::string path, std::string name) {
	std::ifstream file(path + "/" + name, std::ios_base::in);

	long firstIndex = -1;

	if(file.is_open()) {
		std::string line;
		bool readingControllerData = false;

		while(getline(file, line)) {
			// Check if we can/continue reading controller data
			std::size_t pos = line.find("#BEGIN");
			if(pos != std::string::npos) {
				line = line.substr(7);
				pos = line.find(" ");
				if(pos != std::string::npos) {
					readingControllerData = true;
					continue;
				}	
			}

			pos = line.find("#END");
			if(pos != std::string::npos && readingControllerData) {
				readingControllerData = false;
				continue;
			}

			// Process controller data
			if(readingControllerData) {
				pos = line.find(" ");

				unsigned long stateIndex = std::stol(line.substr(0, pos));
				unsigned long inputIndex = std::stol(line.substr(pos + 1));

				controller[stateIndex] = inputIndex;

				if(firstIndex == -1) firstIndex = stateIndex;
			}	
		}
	}
	file.close();

	return firstIndex;
}


// Converts an index into a boolean expression
std::string indexToBoolean(unsigned long index, unsigned int wordLength) {
	std::string boolean(wordLength, '0');

	for(int i = wordLength - 1; i >= 0; i--) {
		auto denominator = pow(2.0, (double)i);

		if ((index - denominator) >= 0) {
			boolean[i] = '1';
			index -= denominator;
		} else {
			boolean[i] = '0';
		}
	}

	return boolean;
}


// Converts a boolean into an index
unsigned long booleanToIndex(std::string boolean) {
	unsigned long value = 0;
	for(unsigned int i = 0; i < boolean.length(); i++) {
		if(boolean[i] == '1') {
			value += pow(2.0, (double)i);
		}
	}

	return value;
}


// Computes a BDD representation of a boolean string
BDD computeBooleanRepresentation(Cudd* cudd, BDD* vars, std::string word) {
	unsigned int length = word.length();

	BDD representation = cudd->bddOne();
	for(unsigned int i = 0; i < length; i++) {
		if(word[i] == '1') representation = representation & vars[i];
		else representation = representation & !vars[i];
	}

	return representation;
}


// Saves a BDD
void saveBdd(std::string name, Cudd* cudd, BDD* bdd) {
	std::string path =  "controllers/" + name + ".bdd";
	char cpath[path.size() + 1];
	strcpy(cpath, path.c_str());

	FILE* file = fopen(cpath, "w");

	int storeStatus = Dddmp_cuddBddStore(
		cudd->getManager(),
		NULL,
		bdd->getNode(),
		NULL, 
		NULL,
		DDDMP_MODE_BINARY,
		DDDMP_VARIDS,
		NULL,
		file
    );

    if(storeStatus != DDDMP_SUCCESS) {
    	std::cout << "Failed to save BDD" << std::endl;
    } else {
    	std::cout << "BDD succesfully saved" << std::endl;
    }

	fclose(file);
}


int main() {
	std::cout << "COSYNNC: BDD generator tool" << std::endl << std::endl;
	string filename = "staticController.scs";

	Cudd cudd;

	// Enable variable reordering
	cudd.AutodynEnable();

	// Read the static controller
	unsigned long* staticController = new unsigned long[stateCardinality];
	for(unsigned long index = 0; index < stateCardinality; index++) staticController[index] = ULONG_MAX;
	long firstIndex = readStaticController(staticController, "controllers", filename);

	// Generate BDD variables
	BDD *stateVars = new BDD[stateWordLength];
	for(unsigned int i = 0; i < stateWordLength; i++) stateVars[i] = cudd.bddVar(i);

	BDD *inputVars = new BDD[inputWordLength];
	for(unsigned int i = 0; i < inputWordLength; i++) inputVars[i] = cudd.bddVar(stateWordLength + i);

	BDD controller = cudd.bddZero();
	BDD winningSet = cudd.bddZero();

	// Generate the BDD representation of the controller
	for(unsigned long index = firstIndex; index < stateCardinality; index++) {
		if(staticController[index] == ULONG_MAX) continue;

		auto inputIndex = staticController[index];
		if(inputIndex > inputCardinality) continue;

		auto stateBoolean = indexToBoolean(index, stateWordLength);
		auto stateBdd = computeBooleanRepresentation(&cudd, stateVars, stateBoolean);

		auto inputBoolean = indexToBoolean(inputIndex, inputWordLength);
		auto inputBdd = computeBooleanRepresentation(&cudd, inputVars, inputBoolean);

		auto relationBdd = stateBdd & inputBdd;

		controller = controller | relationBdd;
		winningSet = winningSet | stateBdd;
	}
	std::cout << "Conversion successful" << std::endl << std::endl;

	// Attempt to reduce the heap of the controller
	std::cout << "Node count: " << cudd.ReadNodeCount() << std::endl;
	Cudd_ReduceHeap(cudd.getManager(), CUDD_REORDER_SIFT, 1);
	std::cout << "Node count: " << cudd.ReadNodeCount() << std::endl << std::endl;

	// Save the BDD for data comparison purposes
	saveBdd("controller", &cudd, &controller);
	saveBdd("winningSet", &cudd, &winningSet);

	// Free up memory 
	delete[] staticController;

	delete[] stateVars;
	delete[] inputVars;
}