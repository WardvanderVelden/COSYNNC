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

		// Constructor that initializes the filemanager for use with control goal and controller included
		FileManager(NeuralNetwork* neuralNetwork, Verifier* verifier, Quantizer* stateQuantizer, Quantizer* inputQuantizer, ControlSpecification* specification, Controller* controller);

		// Loads a neural network
		void LoadNetworkFromMATLAB(string path, string name);

		// Save network in its binary form to yield the most compressed representation
		void SaveNetwork(string path, string name);

		// Save the structure of a neural network to a MATLAB file
		void SaveNetworkAsMATLAB(string path, string name);

		// Save the verified domain to a MATLAB file
		void SaveVerifiedDomainAsMATLAB(string path, string name);

		// Save the controller as a static controller, just like old versions of SCOTS used to do
		void SaveControllerAsStaticController(string path, string name);

		// Writes the synthesis status to the log file for debug purposes
		void WriteSynthesisStatusToLog(string path, string name, string plantName, string timestamp);

		// Writes the quantization parameters for the state and input quantizer to a MATLAB file
		void WriteQuantizationParametersToMATLABFile(ofstream* file);

		// Write the quantization parameters for the state and input quantizer to a static controller file
		void WriteQuantizationParametersToStaticController(ofstream* file);

		// Writes a vector to a file
		void WriteVectorToMATLABFile(ofstream* file, string variableName, Vector vector);

		// Writes a vector in static controller SCOTS format
		void WriteVectorToStaticController(ofstream* file, string variableName, Vector vector);

		// Writes a value as bytes into a file
		void WriteByte(ofstream* file, unsigned char value);
		void WriteIntAsBytes(ofstream* file, int value);
		void WriteFloatAsBytes(ofstream* file, float value);
	private:
		NeuralNetwork* _neuralNetwork = nullptr;
		Verifier* _verifier = nullptr;

		Quantizer* _stateQuantizer = nullptr;
		Quantizer* _inputQuantizer = nullptr;

		ControlSpecification* _specification = nullptr;

		Controller* _controller = nullptr;
	};
}
