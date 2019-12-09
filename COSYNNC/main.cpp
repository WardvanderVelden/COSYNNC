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
	const float _mass = 267; // kg
	const float _g = -9.81; // m s^-2
}; 

int main() {
	std::cout << "COSYNNC: A correct-by-design neural network synthesis tool." << std::endl << std::endl;

	
	// COSYNNC training parameters
	const int episodes = 2500000;
	const int steps = 25;
	const int verboseEpisode = 2500;

	// Initialize quantizers
	Quantizer* stateQuantizer = new Quantizer(true);
	stateQuantizer->SetQuantizeParameters(Vector({ 0.1, 0.1 }), Vector({ -5, -10 }), Vector({ 5, 10 }));

	Quantizer* inputQuantizer = new Quantizer(true);
	inputQuantizer->SetQuantizeParameters(Vector((float)5000.0), Vector((float)-2500.0), Vector((float)7500.0));


	// Initialize plant
	Rocket* plant = new Rocket();


	// Initialize controller
	Controller controller(plant, stateQuantizer, inputQuantizer);


	// Initialize a control specification
	ControlSpecification specification(ControlSpecificationType::Reachability, plant);
	specification.SetHyperInterval(Vector({ -1, -1 }), Vector({ 1, 1 }));
	controller.SetControlSpecification(&specification);


	// Initialize a multilayer perceptron neural network and configure it to function as controller
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 32, 32 }, ActivationActType::kRelu);
	multilayerPerceptron->InitializeOptimizer("adam", 0.0075, 0.001, false); // 32 32 (or 16 16 16 16)
	multilayerPerceptron->ConfigurateInputOutput(plant, inputQuantizer, steps, 1.0);
	controller.SetNeuralNetwork(multilayerPerceptron);


	// TEMPORARY: Proof of concept for the reinforcement learning synthesis routine
	std::cout << "Training" << std::endl;
	for (int i = 0; i < episodes; i++) {
		if (i % verboseEpisode == 0) std::cout << std::endl;

		vector<Vector> states;
		vector<Vector> reinforcingLabels;
		vector<Vector> deterringLabels;

		auto initialState = stateQuantizer->GetRandomVector();
		initialState[0] = initialState[0] * 0.75;
		initialState[1] = initialState[1] * 0.35;
		plant->SetState(initialState);

		vector<float> normWeights = { 1.0, 0.0 };
		auto initialNorm = (initialState - specification.GetCenter()).GetWeightedNorm(normWeights);
		auto oldNorm = initialNorm;

		bool isInStateSet = true;
		bool isAtGoal = false;

		// Simulate the episode using the neural network
		for (int j = 0; j < steps; j++) {
			auto normalizedQuantizedState = stateQuantizer->NormalizeVector(stateQuantizer->QuantizeVector(plant->GetState()));

			// Get network output for the quantized state
			auto networkOutput = multilayerPerceptron->EvaluateNetwork(normalizedQuantizedState);

			auto amountOfLabels = networkOutput.GetLength();

			// Get total probability of all the labels
			float totalProbability = 0.0;
			for (int i = 0; i < amountOfLabels; i++) totalProbability += networkOutput[i];

			// Normalize probabilities
			for (int i = 0; i < amountOfLabels; i++) networkOutput[i] = networkOutput[i] / totalProbability;

			// Determine a one hot vector from the network output
			float randomValue = ((float)rand() / RAND_MAX);
			float cumalitiveProbability = 0.0;
			int oneHotIndex = 0;
			for (int i = 0; i < amountOfLabels; i++) {
				cumalitiveProbability += networkOutput[i];
				if (randomValue < cumalitiveProbability) {
					oneHotIndex = i;
					break;
				}
			}

			// Determine input from one hot vector
			Vector oneHot(networkOutput.GetLength());
			oneHot[oneHotIndex] = 1.0;

			auto input = inputQuantizer->FindVectorFromOneHot(oneHot);


			// Evolve the plant
			plant->Evolve(input);
			auto state = plant->GetState();

			// See if the evolved state is at the control goal and if the evolved state is still in the state set
			auto norm = (state - specification.GetCenter()).GetWeightedNorm(normWeights);

			isAtGoal = specification.IsInControlGoal(state);
			isInStateSet = stateQuantizer->IsInBounds(state);

			// Find the reinforcing labels and the deterring labels
			Vector reinforcementLabel = Vector(networkOutput.GetLength());
			Vector deterringLabel = Vector(networkOutput.GetLength());

			reinforcementLabel = oneHot;
			for (int i = 0; i < amountOfLabels; i++) deterringLabel[i] = 1.0 - reinforcementLabel[i];

			oldNorm = norm;

			// Add states and labels to the list of states and labels
			states.push_back(normalizedQuantizedState);
			reinforcingLabels.push_back(reinforcementLabel);
			deterringLabels.push_back(deterringLabel);

			// DEBUG: Print simulation for verification purposes
			if (i % verboseEpisode == 0) std::cout << "i: " << i << "\tj: " << j << "\t\tx0: " << state[0] << "\tx1: " << state[1] << "\t\tp0: " << networkOutput[0] << "\tp1: " << networkOutput[1] << "\tu: " << input[0] << "\ts: " << isAtGoal << std::endl;

			// Stop episode if we are in the goal state or have left the bounded set
			if (isAtGoal || !isInStateSet) break;
		}

		// Train the neural network based on the performance of the network
		if (oldNorm < initialNorm || isAtGoal) multilayerPerceptron->Train(states, reinforcingLabels); // Reinforce if we are closer to the goal
		else multilayerPerceptron->Train(states, deterringLabels);

		//if (isAtGoal) multilayerPerceptron->Train(states, reinforcingLabels); // Reinforce if we are closer to the goal
		//else multilayerPerceptron->Train(states, deterringLabels);
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