#include "NeuralNetwork.h"

namespace COSYNNC {
	NeuralNetwork::NeuralNetwork() {

		
	}


	// Initialize the neural network topology
	void NeuralNetwork::InitializeNetworkTopology() {

	}

	// Evaluates the neural network
	Vector NeuralNetwork::EvaluateNetwork(Vector input) {
		return Vector((float)0);
	}

	// Sets the layers of the network (if we consider a fully connected network topology)
	void NeuralNetwork::SetLayers(vector<int> layers) {
		_layers = layers;
		_depth = layers.size();
	}
}