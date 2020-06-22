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

		// Constructor that initializes the filemanager
		FileManager(NeuralNetwork* neuralNetwork, Verifier* verifier, Abstraction* abstraction);

		// Loads a neural network
		void LoadNetworkFromMATLAB(string path, string name);

		// Loads a static controller returning a controller object that contains the deterministic static controllers behaviour
		Controller* LoadStaticController(string path, string name);

		// Save network in its binary form to yield the most compressed representation
		void SaveNetworkAsRaw(string path, string name);

		/// <summary>Save the structure of a neural network to a MATLAB file</summary>
		void SaveNetworkAsMATLAB(string path, string name);

		/// <summary>Save the neural network to a MATLAB file</summary>
		/// <param name="neuralNetwork">A pointer to the neural network that is to be saved</param>
		void SaveNetworkAsMATLAB(string path, string name, NeuralNetwork* neuralNetwork, Controller* controller);

		// Save the winning set to a MATLAB file
		void SaveWinningSetAsMATLAB(string path, string name);

		// Saves the raw controller
		void SaveControllerAsMATLAB(string path, string name);

		// Save the controller as a static controller, just like old versions of SCOTS used to do
		void SaveControllerAsStaticController(string path, string name);

		// Saves the transitions of the plant as known to the abstraction
		void SaveTransitions(string path, string name);

		// Writes the synthesis status to the log file for debug purposes
		void WriteSynthesisStatusToLog(string path, string name, string plantName, string timestamp);	
	private:
		// Writes the quantization parameters for the state and input quantizer to a MATLAB file
		void WriteQuantizationParametersToMATLABFile(ofstream* file);

		// Writes the quantization parameters for the state and input quantizer to a MATLAB file
		void WriteQuantizationParametersToMATLABFile(ofstream* file, Controller* controller);

		// Write the quantization parameters for the state and input quantizer to a static controller file
		void WriteQuantizationParametersToStaticController(ofstream* file);

		// Write the quantization parameters to the transition file
		void WriteQuantizationParametersToAbstractionFile(ofstream* file);

		// Writes a vector to a file
		void WriteVectorToMATLABFile(ofstream* file, string variableName, Vector vector);

		// Writes a vector in static controller SCOTS format
		void WriteVectorToStaticController(ofstream* file, string variableName, Vector vector);

		// Writes a value as bytes into a file
		void WriteByte(ofstream* file, unsigned char value);
		void WriteIntAsBytes(ofstream* file, int value);
		void WriteFloatAsBytes(ofstream* file, float value);

		// Format vector of axis indices into an appropriate string
		string FormatAxisIndices(vector<unsigned long> axisIndices);

		// Format a vector of parameters into a quantizer for the static controller load function
		Quantizer* FormatIntoQuantizer(unsigned int dimension, vector<double> parameters, unsigned int significance = 4);

		// Rounds a value to a double with a given significance
		double RoundToSignificance(double value, int significance, bool down = false);


		// Pointers to relevant objects
		NeuralNetwork* _neuralNetwork = nullptr;
		Verifier* _verifier = nullptr;
		Abstraction* _abstraction = nullptr;
	};
}
