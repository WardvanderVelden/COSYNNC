#include "FileManager.h"

namespace COSYNNC {
	// Default constructor
	FileManager::FileManager() { }


	// Constructor that initializes the filemanager
	FileManager::FileManager(NeuralNetwork* neuralNetwork, Verifier* verifier, Abstraction* abstraction) {
		_neuralNetwork = neuralNetwork;
		_verifier = verifier;
		_abstraction = abstraction;
	}


	// Loads a neural network
	void FileManager::LoadNetworkFromMATLAB(string path, string name) {
		ifstream file(path + "/" + name, std::ios_base::in);

		if (file.is_open()) {
			StringHelper stringHelper;

			string line;
			
			string currentArgument = "";
			bool readingArgument = false;
			vector<mx_float> data;

			while (getline(file, line)) {
				// Weight information
				if (line[0] == 'w' || line[0] == 'b') {
					if (line.find(' ') > 5) continue;

					auto firstSpace = line.find(' ');
					currentArgument = line.substr(0, firstSpace);
					line = line.substr(firstSpace + 1, line.size() - firstSpace - 1);

					readingArgument = true;
				}

				if (readingArgument) {
					if (line.find(';') != -1) readingArgument = false;

					stringHelper.ReplaceAll(line, '\t');
					stringHelper.ReplaceAll(line, '\n');
					stringHelper.ReplaceAll(line, '=');
					stringHelper.ReplaceAll(line, ';');
					stringHelper.ReplaceAll(line, '[');
					stringHelper.ReplaceAll(line, ']');
					stringHelper.ReplaceAll(line, ' ');
					
					auto vec = stringHelper.Split(line, ',');

					if (vec.size() > 0) {
						for (unsigned int i = 0; i < vec.size(); i++) data.push_back(stof(vec[i]));
					}

					if(!readingArgument) {
						_neuralNetwork->SetArgument(currentArgument, data);
						data.clear();
					}
				}
			}
		}
		file.close();
	}


	// Save network in its binary form to yield the most compressed representation
	void FileManager::SaveNetworkAsRaw(string path, string name) {
		ofstream file(path + "/" + name + ".nn", std::ios_base::binary);

		// Writes the layer depth with counts the amount of layers minus the input layer
		WriteByte(&file, (unsigned char)_neuralNetwork->GetLayerDepth());

		// Writes the amount of nodes per layer so the network structure is emplied
		WriteByte(&file, (unsigned char)_neuralNetwork->GetInputDimension());

		auto layers = _neuralNetwork->GetLayers();
		for (unsigned int i = 0; i < _neuralNetwork->GetLayerDepth(); i++) {
			WriteByte(&file, (unsigned char)layers[i]);
		}

		// Writes all the weights and biases to the file
		auto argumentNames = _neuralNetwork->GetArgumentNames();
		for (unsigned int i = 0; i < argumentNames.size(); i++) {
			auto argumentName = argumentNames[i];
			if (argumentName == "input" || argumentName == "label") continue;
			auto argumentShape = _neuralNetwork->GetArgumentShape(argumentName);

			// Vector
			if (argumentShape.size() == 1) {
				for (unsigned int i = 0; i < argumentShape[0]; i++) {
					WriteFloatAsBytes(&file, _neuralNetwork->GetArgumentValue(argumentName, vector<unsigned int>({ i })));
				}
			}
			// Matrix
			else {
				for (unsigned int i = 0; i < argumentShape[0]; i++) {
					for (unsigned int j = 0; j < argumentShape[1]; j++) {
						WriteFloatAsBytes(&file, _neuralNetwork->GetArgumentValue(argumentName, vector<unsigned int>({ i, j })));
					}
				}
			}
		}

		// Close file
		file.close();
	}


	// Save the structure of a neural network to a MATLAB file 	// TODO: Adapt this depending on which type of neural network is currently being trained
	void FileManager::SaveNetworkAsMATLAB(string path, string name) {
		ofstream file(path + "/" + name + ".m", std::ios_base::out);

		// Save the winning domain percentage
		file << "winningDomainPercentage = " << _verifier->GetWinningSetPercentage() << ";\n\n";

		// Save output type of the network
		file << "outputType = '";
		switch (_neuralNetwork->GetOutputType()) {
		case OutputType::Labelled: file << "labelled"; break;
		case OutputType::Range: file << "range"; break;
		}
		file << "';\n";

		// Save activation function
		file << "activationFunction = 'relu';\n"; // TODO: Make this change based on the activation function

		// Save depth
		file << "layerDepth = " << _neuralNetwork->GetLayerDepth() << ";\n";

		// Save the quantization parameters to the network
		WriteQuantizationParametersToMATLABFile(&file);

		// Save the arguments of the network
		file << "\n";
		auto argumentNames = _neuralNetwork->GetArgumentNames();
		for (unsigned int i = 0; i < argumentNames.size(); i++) {
			auto argumentName = argumentNames[i];
			if (argumentName == "input" || argumentName == "label") continue;

			auto argumentShape = _neuralNetwork->GetArgumentShape(argumentName);

			file << argumentName << " = [";

			// Vector
			if (argumentShape.size() == 1) {
				for (unsigned int i = 0; i < argumentShape[0]; i++) {
					file << _neuralNetwork->GetArgumentValue(argumentName, vector<unsigned int>({ i }));
					if (i != argumentShape[0] - 1) file << ", ";
				}
			}
			// Matrix
			else {
				file << "\n";
				for (unsigned int i = 0; i < argumentShape[0]; i++) {
					file << "\t[";
					for (unsigned int j = 0; j < argumentShape[1]; j++) {
						file << _neuralNetwork->GetArgumentValue(argumentName, vector<unsigned int>({ i, j }));
						if (j != (argumentShape[1] - 1)) file << ", ";
					}
					file << "],\n";
				}
			}
			file << "];\n";
		}
		file.close();
	}


	// Save the verified domain to a MATLAB file
	void FileManager::SaveVerifiedDomainAsMATLAB(string path, string name) {
		ofstream file(path + "/" + name + ".m", std::ios_base::out);

		// Save the winning domain to the domain file
		file << "winningDomainPercentage = " << _verifier->GetWinningSetPercentage() << ";\n";
		file << "networkDataSize = " << _neuralNetwork->GetDataSize() << ";\n";

		// Save the quantization parameters to the domain file
		WriteQuantizationParametersToMATLABFile(&file);

		// Save the controller goal which the domain describes
		if (_abstraction->GetControlSpecification() != nullptr) {
			file << "\n";
			WriteVectorToMATLABFile(&file, "goalLowerVertex", _abstraction->GetControlSpecification()->GetLowerHyperIntervalVertex());
			WriteVectorToMATLABFile(&file, "goalUpperVertex", _abstraction->GetControlSpecification()->GetUpperHyperIntervalVertex());
		}

		// Save the winning domain
		unsigned long stateCardinality = _abstraction->GetStateQuantizer()->GetCardinality();

		file << "\nwinningDomain = zeros(" << stateCardinality << ", 1);\n\n";
		for (unsigned long index = 0; index < stateCardinality; index++) {
			auto value = (_verifier->IsIndexInWinningSet(index)) ? 1 : 0;
			file << "domain(" << (index + 1) << ") = " << value << ";\n";
		}

		file.close();
	}


	// Save the controller as a static controller, just like old versions of SCOTS used to do
	void FileManager::SaveControllerAsStaticController(string path, string name) {
		ofstream file(path + "/" + name + ".scs", std::ios_base::out);

		file << "#SCOTS:v0.2\n#TYPE:STATICCONTROLLER\n";

		WriteQuantizationParametersToStaticController(&file);

		file << "#TYPE:WINNINGDOMAIN\n#MATRIX:DATA\n";
		file << "#BEGIN:" << _verifier->GetWinningSetSize() << " " << _abstraction->GetInputQuantizer()->GetCardinality() << "\n";

		// State inputs that the neural network gives for every state
		const auto _spaceCardinality = _abstraction->GetStateQuantizer()->GetCardinality();

		for (unsigned long index = 0; index < _spaceCardinality; index++) {
			if (_verifier->IsIndexInWinningSet(index)) {
				auto input = _abstraction->GetController()->GetControlActionFromIndex(index);
				auto inputIndex = _abstraction->GetInputQuantizer()->GetIndexFromVector(input);

				file << index << " " << inputIndex << "\n";
			}
		}
		file << "#END";

		file.close();
	}


	// Writes the synthesis status to the log file for debug purposes
	void FileManager::WriteSynthesisStatusToLog(string path, string name, string plantName, string timestamp) {
		ofstream file(path + "/" + name + ".txt", std::ios_base::app);

		file <<  plantName << " " << timestamp << " " << _verifier->GetWinningSetPercentage() << "\n";

		file.close();
	}


	// Writes the quantization parameters for the state and input quantizer to the file
	void FileManager::WriteQuantizationParametersToMATLABFile(ofstream* file) {
		auto stateDimension = _abstraction->GetStateQuantizer()->GetDimension();

		// Save state space quantization parameters
		*file << "\n";
		WriteVectorToMATLABFile(file, "stateSpaceEta", _abstraction->GetStateQuantizer()->GetEta());
		WriteVectorToMATLABFile(file, "stateSpaceLowerBound", _abstraction->GetStateQuantizer()->GetLowerBound());
		WriteVectorToMATLABFile(file, "stateSpaceUpperBound", _abstraction->GetStateQuantizer()->GetUpperBound());

		// Save input space quantization parameters
		*file << "\n";
		WriteVectorToMATLABFile(file, "inputSpaceEta", _abstraction->GetInputQuantizer()->GetEta());
		WriteVectorToMATLABFile(file, "inputSpaceLowerBound", _abstraction->GetInputQuantizer()->GetLowerBound());
		WriteVectorToMATLABFile(file, "inputSpaceUpperBound", _abstraction->GetInputQuantizer()->GetUpperBound());
	}


	// Write the quantization parameters for the state and input quantizer to a static controller file
	void FileManager::WriteQuantizationParametersToStaticController(ofstream* file) {
		*file << "SCOTS:STATE_SPACE\n#TYPE:UNIFORMGRID\nMEMBER:DIM\n";
		*file << _abstraction->GetStateQuantizer()->GetDimension() << "\n";

		WriteVectorToStaticController(file, "ETA", _abstraction->GetStateQuantizer()->GetEta());
		WriteVectorToStaticController(file, "LOWER_LEFT", _abstraction->GetStateQuantizer()->GetLowerBound());
		WriteVectorToStaticController(file, "UPPER_RIGHT", _abstraction->GetStateQuantizer()->GetUpperBound());

		*file << "#SCOTS:INPUT_SPACE\n#TYPE:UNIFORMGRID\nMEMBER:DIM\n";
		*file << _abstraction->GetInputQuantizer()->GetDimension() << "\n";

		WriteVectorToStaticController(file, "ETA", _abstraction->GetInputQuantizer()->GetEta());
		WriteVectorToStaticController(file, "LOWER_LEFT", _abstraction->GetInputQuantizer()->GetLowerBound());
		WriteVectorToStaticController(file, "UPPER_RIGHT", _abstraction->GetInputQuantizer()->GetUpperBound());
	}


	// Writes a vector to a file
	void FileManager::WriteVectorToMATLABFile(ofstream* file, string variableName, Vector vector) {
		*file << variableName << " = [";
		for (unsigned int i = 0; i < vector.GetLength(); i++) {
			*file << vector[i];
			if (i != (vector.GetLength() - 1)) *file << ", ";
		}
		*file << "];\n";
	}


	// Writes a vector in static controller SCOTS format
	void FileManager::WriteVectorToStaticController(ofstream* file, string variableName, Vector vector) {
		*file << "#VECTOR:" << variableName << "\n";
		*file << "#BEGIN:" << vector.GetLength() << "\n";

		for (unsigned int i = 0; i < vector.GetLength(); i++) {
			*file << vector[i] << "\n";
		}
		*file << "#END\n";
	}


	// Writes a byte into a file
	void FileManager::WriteByte(ofstream* file, unsigned char value) {
		unsigned char* b = (unsigned char*)&value;

		for (unsigned int i = 0; i < sizeof(value); i++) {
			*file << b[i];
		}
	}


	// Writes an int as bytes into a file
	void FileManager::WriteIntAsBytes(ofstream* file, int value) {
		unsigned char* b = (unsigned char*)&value;

		for (unsigned int i = 0; i < sizeof(value); i++) {
			*file << b[i];
		}
	}


	// Writes a float as bytes into a file
	void FileManager::WriteFloatAsBytes(ofstream* file, float value) {
		unsigned char *b = (unsigned char *)&value;

		for (unsigned int i = 0; i < sizeof(value); i++) {
			*file << b[i];
		}
	}
}