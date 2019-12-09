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
	struct TrainingData {
		NDArray inputs;
		NDArray labels;
	};

	class MultilayerPerceptron : public NeuralNetwork  
	{
	public:
		// Initializes a neural network of the multilayer perceptron topology, the layers exclude the input and the output (infered from the plant)
		MultilayerPerceptron(vector<int> hiddenLayers, ActivationActType activationFunction);

		// Initializes a multilayer perceptron neural network topology
		virtual void InitializeGraph();

		// DEBUG: Temporay test bed for learning MXNET
		void Test(TrainingData* data, Quantizer* stateQuantizer, Quantizer* inputQuantize);

		// DEBUG: Get test data
		//TrainingData* GetTrainingData(Plant* plant, Controller* controller, Quantizer* stateQuantizer, Quantizer* inputQuantizer);
	private:
		vector<Symbol> _weights;
		vector<Symbol> _biases;
		vector<Symbol> _outputs;

		ActivationActType _activationFunction;
	};
}


