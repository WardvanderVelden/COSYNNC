#include "FileManager.h"

namespace COSYNNC {
	// Default constructor
	FileManager::FileManager() { }


	// Constructor that initializes the file manager for use
	FileManager::FileManager(NeuralNetwork* neuralNetwork, Verifier* verifier, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		_neuralNetwork = neuralNetwork;
		_verifier = verifier;

		_stateQuantizer = stateQuantizer;
		_inputQuantizer = inputQuantizer;
	}

	// Constructor that initializes the file manager for use
	FileManager::FileManager(NeuralNetwork* neuralNetwork, Verifier* verifier, Quantizer* stateQuantizer, Quantizer* inputQuantizer, ControlSpecification* specification) {
		_neuralNetwork = neuralNetwork;
		_verifier = verifier;

		_stateQuantizer = stateQuantizer;
		_inputQuantizer = inputQuantizer;

		_specification = specification;
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


	// Save the structure of a neural network to a MATLAB file
	// TODO: Adapt this depending on which type of neural network is currently being trained
	void FileManager::SaveNetworkAsMATLAB(string path, string name) {
		ofstream file(path + "/" + name + ".m", std::ios_base::out);

		// Save the winning domain percentage
		file << "winningDomainPercentage = " << _verifier->GetWinningDomainPercentage() << ";\n\n";

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
		WriteQuantizationParametersToFile(&file);

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
		file << "winningDomainPercentage = " << _verifier->GetWinningDomainPercentage() << ";\n";

		// Save the quantization parameters to the domain file
		WriteQuantizationParametersToFile(&file);

		// Save the controller goal which the domain describes
		if (_specification != nullptr) {
			file << "\n";
			WriteVectorToMATLABFile(&file, "goalLowerVertex", _specification->GetLowerHyperIntervalVertex());
			WriteVectorToMATLABFile(&file, "goalUpperVertex", _specification->GetUpperHyperIntervalVertex());
		}

		// Save the winning domain
		unsigned long stateCardinality = _stateQuantizer->GetCardinality();

		file << "\nwinningDomain = zeros(" << stateCardinality << ", 1);\n\n";
		for (unsigned long index = 0; index < stateCardinality; index++) {
			auto value = (_verifier->IsIndexInWinningSet(index)) ? 1 : 0;
			file << "domain(" << (index + 1) << ") = " << value << ";\n";
		}

		file.close();
	}

	// Writes the quantization parameters for the state and input quantizer to the file
	void FileManager::WriteQuantizationParametersToFile(ofstream* file) {
		auto stateDimension = _stateQuantizer->GetSpaceDimension();

		// Save state space quantization parameters
		*file << "\n";
		WriteVectorToMATLABFile(file, "stateSpaceEta", _stateQuantizer->GetSpaceEta());
		WriteVectorToMATLABFile(file, "stateSpaceLowerBound", _stateQuantizer->GetSpaceLowerBound());
		WriteVectorToMATLABFile(file, "stateSpaceUpperBound", _stateQuantizer->GetSpaceUpperBound());

		// Save input space quantization parameters
		*file << "\n";
		WriteVectorToMATLABFile(file, "inputSpaceEta", _inputQuantizer->GetSpaceEta());
		WriteVectorToMATLABFile(file, "inputSpaceLowerBound", _inputQuantizer->GetSpaceLowerBound());
		WriteVectorToMATLABFile(file, "inputSpaceUpperBound", _inputQuantizer->GetSpaceUpperBound());
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
}