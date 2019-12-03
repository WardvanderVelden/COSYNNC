#include <iostream>
#include "Vector.h"
#include "Plant.h"
#include "Quantizer.h"
#include "Controller.h"
#include "ControlSpecification.h"
#include "MultilayerPerceptron.h"

using namespace std;
using namespace mxnet;

class Rocket : public Plant {
public:
	Rocket() : Plant(2, 1, 0.1) { }

	// Simple second order rocket dynamics in one axis
	Vector SingleStepDynamics(Vector input) override {
		Vector newState(GetStateSpaceDimension());

		newState[0] = GetState()[0] + GetState()[1] * GetTau();
		newState[1] = GetState()[1] + GetTau() / _mass * input[0] + _g * GetTau();

		return newState;
	}
private:
	float _mass = 267; // kg
	float _g = -9.81; // m s^-2
}; 

int main() {
	std::cout << "COSYNNC: A correct-by-design neural network synthesis tool." << std::endl << std::endl;

	
	// COSYNNC training parameters
	const int episodes = 10000;
	const int steps = 25;
	const int verboseEpisode = 1000;


	// Initialize quantizers
	Quantizer* stateQuantizer = new Quantizer(true);
	//stateQuantizer->SetQuantizeParameters(Vector({ 0.1, 0.1 }), Vector({ -5, -2.5 }), Vector({ 5, 2.5 }));
	stateQuantizer->SetQuantizeParameters(Vector({ 0.1, 0.1 }), Vector({ -5, -10 }), Vector({ 5, 10 }));

	Quantizer* inputQuantizer = new Quantizer(true);
	//inputQuantizer->SetQuantizeParameters(Vector((float)100.0), Vector((float)0.0), Vector((float)3000.0));
	inputQuantizer->SetQuantizeParameters(Vector((float)1833.0), Vector((float)0.0), Vector((float)5500.0));


	// Initialize plant
	Rocket* plant = new Rocket();


	// Initialize controller
	Controller controller(plant, stateQuantizer, inputQuantizer);


	// Initialize a control specification
	ControlSpecification specification(ControlSpecificationType::Reachability, plant);
	specification.SetHyperInterval(Vector({ -1, -1 }), Vector({ 1, 1 }));
	controller.SetControlSpecification(&specification);


	// Initialize a multilayer perceptron neural network and configure it to function as controller
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 16, 16, 16, 16 }, ActivationActType::kRelu);
	multilayerPerceptron->InitializeOptimizer("adam", 0.0075, 0.001, false); // 32 32 (or 16 16 16 16)
	multilayerPerceptron->ConfigurateInputOutput(plant, steps);
	controller.SetNeuralNetwork(multilayerPerceptron);


	// TEMPORARY: Proof of concept for the reinforcement learning synthesis routine
	std::cout << "Training" << std::endl;
	for (int i = 0; i < episodes; i++) {
		if (i % verboseEpisode == 0) std::cout << std::endl;

		vector<Vector> states;
		vector<Vector> labels;

		auto initialState = stateQuantizer->GetRandomVector();
		initialState[0] = initialState[0] * 0.75;
		initialState[1] = initialState[1] * 0.3;
		plant->SetState(initialState);

		vector<float> normWeights = { 1.0, 100.0 };
		auto oldDifference = (initialState - specification.GetCenter()).GetWeightedNorm(normWeights);

		bool isInStateSet = true;

		// Simulate the episode using the neural network
		for (int j = 0; j < steps; j++) {
			auto normalizedQuantizedState = stateQuantizer->NormalizeVector(stateQuantizer->QuantizeVector(plant->GetState()));
			auto rawInput = multilayerPerceptron->EvaluateNetwork(normalizedQuantizedState);

			// Pick the input based on the certainty of the network
			auto denormalizedQuantizedInput = inputQuantizer->DenormalizeVector(rawInput);
			auto input = inputQuantizer->QuantizeVector(denormalizedQuantizedInput);

			// Evolve the plant
			plant->Evolve(input);
			auto state = plant->GetState();

			if (!stateQuantizer->IsInBounds(state)) {
				isInStateSet = false;
				break;
			}

			// Find label based on performance
			auto isAtGoal = specification.IsInControlGoal(state);

			Vector label(input.GetLength());

			// Control scheme based on control to minimize the weighted norm, it determines whether or not the resulting control action that led to the new state from
			// the previous state was beneficial. If it was the network will reinforce that input from the previous state to that input
			auto difference = (state - specification.GetCenter()).GetWeightedNorm(normWeights);

			if (difference < oldDifference) label = input;
			else {
				if (state[0] < 0) label = Vector((float)4582.5);
				else label = Vector((float)916.5);

				/*auto randomValue = (float)rand() / RAND_MAX;
				label = inputQuantizer->QuantizeVector(inputQuantizer->DenormalizeVector(Vector(randomValue)));*/
			}

			states.push_back(normalizedQuantizedState);
			labels.push_back(inputQuantizer->NormalizeVector(label));

			oldDifference = difference;

			// DEBUG: Print simulation for verification purposes
			if (i % verboseEpisode == 0) {
				//std::cout << "i: " << i << "\tj: " << j << "\t\tx0: " << state[0] << ":" << states[j][0] <<  "\tx1: " << state[1] << ":" << states[j][1] << "\t\tp: " << rawInput[0] << "\tl: " << label[0] << ":" << labels[j][0] << "\tu: " << input[0] << "\ts: " << isAtGoal << std::endl;
				std::cout << "i: " << i << "\tj: " << j << "\t\tx0: " << state[0] << "\tx1: " << state[1] << "\t\tp: " << rawInput[0] << "\tl: " << label[0] << "\tu: " << input[0] << "\ts: " << isAtGoal << std::endl;
			}
		}

		// Train the neural network based on the performance of the network
		multilayerPerceptron->Train(states, labels);
	}

	// Evaluate some random initial positions to see how the network is performing
	std::cout << std::endl << "Evaluating" << std::endl;
	for (int i = 0; i < 10; i++) {
		auto initialState = stateQuantizer->GetRandomVector();
		initialState[0] = initialState[0] * 0.75;
		initialState[1] = initialState[1] * 0.3;
		plant->SetState(initialState);

		std::cout << std::endl;

		for (int j = 0; j < steps; j++) {
			auto normalizedQuantizedState = stateQuantizer->NormalizeVector(stateQuantizer->QuantizeVector(plant->GetState()));
			auto rawInput = multilayerPerceptron->EvaluateNetwork(normalizedQuantizedState);

			auto denormalizedQuantizedInput = inputQuantizer->DenormalizeVector(rawInput);
			auto input = inputQuantizer->QuantizeVector(denormalizedQuantizedInput);

			// Evolve the plant
			plant->Evolve(input);
			auto state = plant->GetState();

			// Print simulation for evaluation
			std::cout << "i: " << i << "\tj: " << j << "\t\tx0: " << state[0] << "\tx1: " << state[1] << "\t\tp: " << denormalizedQuantizedInput[0] << "\tu: " << input[0] << std::endl;
		}
	}

	// Free memory
	delete plant;
	delete stateQuantizer;
	delete inputQuantizer;
	delete multilayerPerceptron;

	MXNotifyShutdown();

	// Wait for an input to stop the program
	system("pause");
	return 0;
}