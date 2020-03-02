#pragma once
#include <fstream>

#include "mxnet-cpp/MxNetCpp.h"

#include "StringHelper.h"
#include "NeuralNetwork.h"
#include "Verifier.h"

using namespace std;
using namespace mxnet;

namespace COSYNNC {
	class FileManager {
	public:
		// Default constructor
		FileManager();

		// Constructor that initializes the file manager for use
		FileManager(NeuralNetwork* neuralNetwork, Verifier* verifier, Quantizer* stateQuantizer, Quantizer* inputQuantizer);

		// Constructor that initializes the filemanager for use with control goal included
		FileManager(NeuralNetwork* neuralNetwork, Verifier* verifier, Quantizer* stateQuantizer, Quantizer* inputQuantizer, ControlSpecification* specification);

		// Loads a neural network
		void LoadNetworkFromMATLAB(string path, string name);

		// Save the structure of a neural network to a MATLAB file
		void SaveNetworkAsMATLAB(string path, string name);

		// Save the verified domain to a MATLAB file
		void SaveVerifiedDomainAsMATLAB(string path, string name);

		// Writes the quantization parameters for the state and input quantizer to the file
		void WriteQuantizationParametersToFile(ofstream* file);

		// Writes a vector to a file
		void WriteVectorToMATLABFile(ofstream* file, string variableName, Vector vector);
	private:
		NeuralNetwork* _neuralNetwork = nullptr;
		Verifier* _verifier = nullptr;

		Quantizer* _stateQuantizer = nullptr;
		Quantizer* _inputQuantizer = nullptr;

		ControlSpecification* _specification = nullptr;
	};
}
