#pragma once
#include <vector>
#include <map>
#include <iostream>
#include "mxnet-cpp/MxNetCpp.h";
#include "NeuralNetwork.h"
#include "Plant.h"
#include "Controller.h"

using namespace mxnet::cpp;

namespace COSYNNC {
	class MultilayerPerceptron : public NeuralNetwork  
	{
	public:
		// Initializes a neural network of the multilayer perceptron topology, the layers exclude the input and the output (infered from the plant)
		MultilayerPerceptron(vector<int> hiddenLayers, ActivationActType activationFunction, LossFunctionType lossFunction);

		// Initializes a multilayer perceptron neural network topology
		virtual void InitializeGraph();
	private:
		vector<Symbol> _weights;
		vector<Symbol> _biases;
		vector<Symbol> _outputs;

		ActivationActType _activationFunction;
	};
}


