#pragma once
#include "FileManager.h"
#include "Quantizer.h"

namespace COSYNNC {
	class Encoder {
	public:
		// Initializes the encoder based on a SCOTS style static controller
		Encoder(string path, string name);

		// Initializes the encoder based on an abstraction
		Encoder(Abstraction abstraction);

		// Default destructor
		~Encoder();

		// Train the linked neural network to encode the winning set
		void Train();

		// Compute the fitness of the neural network to determine what part is currently encoded succesfully
		float ComputeFitness();
	private:
		FileManager _fileManager;

		Controller* _controller = nullptr;

		Quantizer* _stateQuantizer = nullptr;
		Quantizer* _inputQuantizer = nullptr;
	};
}