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
	const int episodes = 50000;
	const int steps = 15;
	const int verboseEpisode = 1000;


	// Initialize quantizers
	Quantizer* stateQuantizer = new Quantizer(true);
	stateQuantizer->SetQuantizeParameters(Vector({ 0.1, 0.1 }), Vector({ -5, -2.5 }), Vector({ 5, 2.5 }));

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
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 16, 16 }, ActivationActType::kRelu);
	multilayerPerceptron->InitializeOptimizer("adam", 0.05, 0.001);
	multilayerPerceptron->ConfigurateInputOutput(plant, steps);
	controller.SetNeuralNetwork(multilayerPerceptron);


	// TEMPORARY: Proof of concept for the reinforcement learning synthesis routine
	auto initialState = Vector({ -1.2, 0.2 });
	for (int i = 0; i < episodes; i++) {
		if (i % verboseEpisode == 0) std::cout << std::endl;

		vector<Vector> states;
		//vector<Vector> inputs;
		vector<Vector> labels;

		//auto initialState = stateQuantizer->GetRandomVector();
		auto oldIsAtGoal = specification.IsInControlGoal(initialState);

		// Simulate the episode using the neural network
		plant->SetState(initialState);
		for (int j = 0; j < steps; j++) {
			//auto input = controller.GetControlAction(plant->GetState());
			auto normalizedQuantizedState = stateQuantizer->NormalizeVector(stateQuantizer->QuantizeVector(plant->GetState()));
			auto rawInput = multilayerPerceptron->EvaluateNetwork(normalizedQuantizedState);
			
			auto denormalizedQuantizedInput = inputQuantizer->DenormalizeVector(rawInput);
			auto input = inputQuantizer->QuantizeVector(denormalizedQuantizedInput);

			// Evolve the plant
			plant->Evolve(input);
			auto state = plant->GetState();

			// Find label based on performance
			auto isAtGoal = specification.IsInControlGoal(state);

			Vector label(input.GetLength());
			if (isAtGoal) label[0] = 2750.0;
			else if (state[0] < -1 && state[1] < 1) label[0] = 4990;
			else label[0] = 10;

			oldIsAtGoal = isAtGoal;

			// DEBUG: Print simulation for verification purposes
			if (i % verboseEpisode == 0)
				std::cout << "i: " << i << "\tj: " << j << "\tx0: " << state[0] << "\tx1: " << state[1] << "\tp: " << rawInput[0] << "\tu: " << input[0] << "\ts: " << isAtGoal << "\tl: " << label[0] << std::endl;
		
			// Add to history for training after the episode
			states.push_back(stateQuantizer->NormalizeVector(state));
			//inputs.push_back(inputQuantizer->NormalizeVector(input));
			labels.push_back(inputQuantizer->NormalizeVector(label));
		}

		// Train the neural network based on the performance of the network
		multilayerPerceptron->Train(states, labels);
	}


	// Free memory
	delete plant;
	delete stateQuantizer;
	delete inputQuantizer;
	delete multilayerPerceptron;


	// Wait for an input to stop the program
	system("pause");
	return 0;
}