#include "Encoder.h"

namespace COSYNNC {
	// Initializes the encoder based on a SCOTS style static controller
	Encoder::Encoder(string path, string name) {
		_fileManager = FileManager();

		_controller = _fileManager.LoadStaticController(path, name);

		_stateQuantizer = _controller->GetStateQuantizer();
		_inputQuantizer = _controller->GetInputQuantizer();
	}

	// Initializes the encoder based on an abstraction
	Encoder::Encoder(Abstraction abstraction) {

	}

	// Default destructor
	Encoder::~Encoder() {
		delete _stateQuantizer;
		delete _inputQuantizer;
	}

	// Train the linked neural network to encode the winning set
	void Encoder::Train() {

	}

	// Compute the fitness of the neural network to determine what part is currently encoded succesfully
	float Encoder::ComputeFitness() {
		return 0.0;
	}
}