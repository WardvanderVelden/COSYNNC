#pragma once
#include <vector>
#include <map>
#include "mxnet-cpp/MxNetCpp.h";
#include "NeuralNetwork.h"

using namespace mxnet::cpp;

namespace COSYNNC {
	class MultilayerPerceptron : public NeuralNetwork  
	{
	public:
		// Constructs and initializes a multilayer perceptron neural network topology
		MultilayerPerceptron(vector<int> layers, ActivationActType activationFunction);

		// Initializes a multilayer perceptron neural network topology
		virtual void InitializeNetworkTopology();

		// DEBUG: Temporay test bed for learning MXNET
		void Test();
	private:
		vector<Symbol> _weights;
		vector<Symbol> _biases;
		vector<Symbol> _outputs;

		ActivationActType _activationFunction;
	};
}


