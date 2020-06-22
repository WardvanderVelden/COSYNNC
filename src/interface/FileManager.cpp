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
		ifstream file(path + "/" + name + ".m", std::ios_base::in);

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

					if (!readingArgument) {
						_neuralNetwork->SetArgument(currentArgument, data);
						data.clear();
					}
				}
			}
		}
		file.close();
	}


	// Loads a static controller returning a controller object that contains the deterministic static controllers behaviour
	Controller* FileManager::LoadStaticController(string path, string name) {
		ifstream file(path + "/" + name + ".scs", std::ios_base::in);

		Controller* controller = nullptr;
		if (file.is_open()) {

			Quantizer* stateQuantizer = nullptr;
			Quantizer* inputQuantizer = nullptr;

			StringHelper stringHelper;
			string line;

			// Argument reading variables
			bool nextLineIsADim = false;
			bool readingArgument = false;
			vector<unsigned int> dimensions;
			vector<double> arguments;

			// Controller reading variables
			bool readingController = false;

			while (getline(file, line)) {
				// End processing arguments
				if (line.find("END") != -1 && !readingController) {
					readingArgument = false;
				}

				// Add argument to the vector of arguments
				if (readingArgument && !readingController) {
					arguments.push_back(stof(line));
				}

				// Start processing arguments
				if (line.find("BEGIN") != -1 && !readingController) {
					readingArgument = true;
				}

				// Check if next line is a dimension
				if (nextLineIsADim) {
					dimensions.push_back(stoi(line));
					nextLineIsADim = false;
				}
				if (line.find("DIM") != -1 && !readingController) nextLineIsADim = true;

				// Check if we start reading the controller if so process previously attained parameters into appropriate quantizers
				if (line.find("WINNINGDOMAIN") != -1) {
					readingController = true;

					vector<double> stateParameters;
					vector<double> inputParameters;

					for (size_t i = 0; i < dimensions[0] * 3; i++) stateParameters.push_back(arguments[i]);
					for (size_t i = 0; i < dimensions[1] * 3; i++) inputParameters.push_back(arguments[dimensions[0] * 3 + i]);

					// Instantiate quantizers
					stateQuantizer = FormatIntoQuantizer(dimensions[0], stateParameters);
					inputQuantizer = FormatIntoQuantizer(dimensions[1], inputParameters);

					// Initialize controller
					controller = new Controller(stateQuantizer, inputQuantizer);
					controller->InitializeInputs();
				}

				if (readingController) {
					if (line.find("#") == -1 && line.find(" ") != -1) {
						auto elements = stringHelper.Split(line, ' ');

						if (elements.size() == 2) {
							unsigned long stateIndex = stol(elements[0]);
							unsigned long inputIndex = stol(elements[1]);

							Vector input = inputQuantizer->GetVectorFromIndex(inputIndex);
							controller->SetInput(stateIndex, input);
						}
					}
				}
			}
		}
		file.close();

		return controller;
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


	// Save the structure of a neural network to a MATLAB file
	void FileManager::SaveNetworkAsMATLAB(string path, string name) {
		ofstream file(path + "/" + name + ".m", std::ios_base::out);
		file.precision(9);

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


	// Save the neural network to a MATLAB file
	void FileManager::SaveNetworkAsMATLAB(string path, string name, NeuralNetwork* neuralNetwork, Controller* controller) {
		ofstream file(path + "/" + name + ".m", std::ios_base::out);
		file.precision(9);

		// Save activation function
		file << "activationFunction = 'relu';\n"; // TODO: Make this change based on the activation function

		// Save depth
		file << "layerDepth = " << neuralNetwork->GetLayerDepth() << ";\n";

		// Save the quantization parameters to the network
		WriteQuantizationParametersToMATLABFile(&file, controller);

		// Save the arguments of the network
		file << "\n";
		auto argumentNames = neuralNetwork->GetArgumentNames();
		for (unsigned int i = 0; i < argumentNames.size(); i++) {
			auto argumentName = argumentNames[i];
			if (argumentName == "input" || argumentName == "label") continue;

			auto argumentShape = neuralNetwork->GetArgumentShape(argumentName);

			file << argumentName << " = [";

			// Vector
			if (argumentShape.size() == 1) {
				for (unsigned int i = 0; i < argumentShape[0]; i++) {
					file << neuralNetwork->GetArgumentValue(argumentName, vector<unsigned int>({ i }));
					if (i != argumentShape[0] - 1) file << ", ";
				}
			}
			// Matrix
			else {
				file << "\n";
				for (unsigned int i = 0; i < argumentShape[0]; i++) {
					file << "\t[";
					for (unsigned int j = 0; j < argumentShape[1]; j++) {
						file << neuralNetwork->GetArgumentValue(argumentName, vector<unsigned int>({ i, j }));
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
	void FileManager::SaveWinningSetAsMATLAB(string path, string name) {
		ofstream file(path + "/" + name + ".m", std::ios_base::out);

		// Save the winning set percentage to the set file
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

		// Save the winning set
		unsigned long stateCardinality = _abstraction->GetStateQuantizer()->GetCardinality();

		file << "\nwinningDomain = zeros(" << stateCardinality << ", 1);\n\n";
		for (unsigned long index = 0; index < stateCardinality; index++) {
			auto value = (_verifier->IsIndexInWinningSet(index)) ? 1 : 0;
			file << "domain(" << (index + 1) << ") = " << value << ";\n";
		}

		file.close();
	}


	// Saves the raw controller
	void FileManager::SaveControllerAsMATLAB(string path, string name) {
		ofstream file(path + "/" + name + ".m", std::ios_base::out);

		auto stateCardinality = _abstraction->GetStateQuantizer()->GetCardinality();
		auto inputDimension = _abstraction->GetInputQuantizer()->GetDimension();
		file << "controller = zeros(" << stateCardinality << ", " << inputDimension << ");\n";
		for (unsigned long index = 0; index < stateCardinality; index++) {
			auto input = _abstraction->GetController()->GetControlActionFromIndex(index);

			file << "controller(" << (index + 1) << ", :) = [";
			for (size_t i = 0; i < inputDimension; i++) {
				file << input[i];
				if (i != (inputDimension - 1)) file << " ";
			}
			file << "];\n";
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


	// Saves the transitions of the plant as known to the abstraction
	void FileManager::SaveTransitions(string path, string name) {
		ofstream file(path + "/" + name + ".trs", std::ios_base::out);
		file.precision(9);

		file << "COSYNNC " << _abstraction->GetPlant()->GetName() << " Abstraction\n";

		WriteQuantizationParametersToAbstractionFile(&file);

		auto spaceEtaR = _abstraction->GetStateQuantizer()->GetEta() * 0.5 * 0.995;

		// Write amount of transitions/ends
		file << "\n";
		if (!_abstraction->IsUsingRefinedTransitions()) file << _abstraction->GetAmountOfTransitions() << "\n";
		else file << _abstraction->GetAmountOfEnds() << "\n";

		// Write all the transitions
		for (unsigned long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
			auto formattedState = FormatAxisIndices(_abstraction->GetStateQuantizer()->GetAxisIndicesFromIndex(index));

			auto transition = _abstraction->GetTransitionOfIndex(index);
			auto processedInputs = transition->GetProcessedInputs();

			for (auto input : processedInputs) {
				auto formattedInput = FormatAxisIndices(_abstraction->GetInputQuantizer()->GetAxisIndicesFromIndex(input));

				auto ends = transition->GetEnds(input);

				if (ends.size() > 0) {
					if (_abstraction->IsUsingRefinedTransitions()) {
						// Output a cell to cell transition that defines the refined transition
						for (auto end : ends) {
							file << formattedState << formattedInput;

							auto newState = _abstraction->GetStateQuantizer()->GetVectorFromIndex(end);
							for (size_t dim = 0; dim < newState.GetLength(); dim++) file << newState[dim] << " ";
							for (size_t dim = 0; dim < newState.GetLength(); dim++) file << spaceEtaR[dim] << " ";
							file << "\n";
						}
					}
					else {
						// Output the hyper rectangular lower and upper bound approximation of the transition
						file << formattedState << formattedInput;
						auto post = transition->GetPost(input);

						auto lowerDiff = transition->GetLowerBound(input) - post;
						auto upperDiff = transition->GetUpperBound(input) - post;

						lowerDiff.Abs(); upperDiff.Abs();
						lowerDiff.Max(upperDiff);

						auto r = lowerDiff;

						// Output x'
						for (unsigned int i = 0; i < post.GetLength(); i++) file << post[i] << " ";

						// Output r
						for (unsigned int i = 0; i < r.GetLength(); i++) file << r[i] << " ";
						file << "\n";
					}
				}
			}
		}

		file.close();
	}


	// Writes the synthesis status to the log file for debug purposes
	void FileManager::WriteSynthesisStatusToLog(string path, string name, string plantName, string timestamp) {
		ofstream file(path + "/" + name + ".txt", std::ios_base::app);

		file << plantName << " " << timestamp << " " << _verifier->GetWinningSetPercentage();
		if (_verifier->GetApparentWinningSetPercentage() > 0.0) file << " " << _verifier->GetApparentWinningSetPercentage();
		file << "\n";

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


	// Writes the quantization parameters for the state and input quantizer to a MATLAB file
	void FileManager::WriteQuantizationParametersToMATLABFile(ofstream* file, Controller* controller) {
		auto stateDimension = controller->GetStateQuantizer()->GetDimension();

		// Save state space quantization parameters
		*file << "\n";
		WriteVectorToMATLABFile(file, "stateSpaceEta", controller->GetStateQuantizer()->GetEta());
		WriteVectorToMATLABFile(file, "stateSpaceLowerBound", controller->GetStateQuantizer()->GetLowerBound());
		WriteVectorToMATLABFile(file, "stateSpaceUpperBound", controller->GetStateQuantizer()->GetUpperBound());

		// Save input space quantization parameters
		*file << "\n";
		WriteVectorToMATLABFile(file, "inputSpaceEta", controller->GetInputQuantizer()->GetEta());
		WriteVectorToMATLABFile(file, "inputSpaceLowerBound", controller->GetInputQuantizer()->GetLowerBound());
		WriteVectorToMATLABFile(file, "inputSpaceUpperBound", controller->GetInputQuantizer()->GetUpperBound());
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


	// Write the quantization parameters to the transition file
	void FileManager::WriteQuantizationParametersToAbstractionFile(ofstream* file) {
		// State space parameters
		auto stateQuantizer = _abstraction->GetStateQuantizer();
		auto stateDimension = stateQuantizer->GetDimension();

		*file << "\n" << stateDimension << "\n";
		for (unsigned int i = 0; i < stateDimension; i++) *file << stateQuantizer->GetLowerBound()[i] << " ";
		*file << "\n";
		for (unsigned int i = 0; i < stateDimension; i++) *file << stateQuantizer->GetUpperBound()[i] << " ";
		*file << "\n";
		for (unsigned int i = 0; i < stateDimension; i++) *file << stateQuantizer->GetEta()[i] << " ";
		*file << "\n";

		// Input space parameters
		auto inputQuantizer = _abstraction->GetInputQuantizer();
		auto inputDimension = inputQuantizer->GetDimension();

		*file << "\n" << inputDimension << "\n";
		for (unsigned int i = 0; i < inputDimension; i++) *file << inputQuantizer->GetLowerBound()[i] << " ";
		*file << "\n";
		for (unsigned int i = 0; i < inputDimension; i++) *file << inputQuantizer->GetUpperBound()[i] << " ";
		*file << "\n";
		for (unsigned int i = 0; i < inputDimension; i++) *file << inputQuantizer->GetEta()[i] << " ";
		*file << "\n";
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


	// Format vector of axis indices into an appropriate string
	string FileManager::FormatAxisIndices(vector<unsigned long> axisIndices) {
		string formattedString = "";

		for (unsigned int i = 0; i < axisIndices.size(); i++) {
			formattedString += to_string(axisIndices[i]) + " ";
		}

		return formattedString;
	}


	// Format a vector of parameters into a quantizer for the static controller load function
	Quantizer* FileManager::FormatIntoQuantizer(unsigned int dimension, vector<double> parameters, unsigned int significance) {
		Vector eta(dimension);
		for (size_t i = 0; i < dimension; i++) eta[i] = RoundToSignificance(parameters[i], significance);

		Vector lowerBound(dimension);
		for (size_t i = 0; i < dimension; i++) lowerBound[i] = RoundToSignificance(parameters[(size_t)(i + dimension)], significance, true);

		Vector upperBound(dimension);
		for (size_t i = 0; i < dimension; i++) upperBound[i] = RoundToSignificance(parameters[(size_t)(i + 2 * dimension)], significance);

		Quantizer* quantizer = new Quantizer();
		quantizer->SetQuantizeParameters(eta, lowerBound, upperBound);

		return quantizer;
	}


	// Rounds a value to a double with a given significance
	double FileManager::RoundToSignificance(double value, int significance, bool down) {
		const unsigned long significanceFactor = pow(10, significance);

		long temporary = 0;
		if(down) temporary = (long)(value * significanceFactor);
		else temporary = (long)(value * significanceFactor + 0.5);

		double newValue = (double)temporary / significanceFactor;

		return newValue;
	}
}