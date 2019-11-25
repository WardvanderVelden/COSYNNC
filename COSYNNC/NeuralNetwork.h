#pragma once
#include "mxnet-cpp/MxNetCpp.h"
#include <vector>
#include "Vector.h"

using namespace mxnet::cpp;

namespace COSYNNC {
	class NeuralNetwork {
	public:
		// Initializes a default neural network
		NeuralNetwork();

		// Initialize the neural network topology
		virtual void InitializeNetworkTopology();

		// Evaluates the neural network
		virtual Vector EvaluateNetwork(Vector input);

		// Sets the layers of the network (if we consider a fully connected network topology)
		void SetLayers(vector<int> layers);

	protected:
		vector<int> _layers;
		int _depth = 0;

		Symbol _network;

		// DEBUG: For now we are running on the cpu
		Context _context = Context::cpu();
	};
}