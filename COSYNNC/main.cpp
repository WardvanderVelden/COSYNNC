#include <iostream>
#include "Vector.h"
#include "Plant.h"
#include "Rocket.h"
#include "Quantizer.h"
#include "Controller.h"
#include "ControlSpecification.h"
#include "MultilayerPerceptron.h"

using namespace std;
using namespace mxnet;

int main() {
	std::cout << "COSYNNC: A correct-by-design neural network synthesis tool." << std::endl << std::endl;

	// COSYNNC training parameters
	const int episodes = 1000000;
	const int steps = 25;
	const int verboseEpisode = 2500;

	// Initialize quantizers
	Quantizer* stateQuantizer = new Quantizer(true);
	stateQuantizer->SetQuantizeParameters(Vector({ 0.1, 0.1 }), Vector({ -5, -10 }), Vector({ 5, 10 }));

	Quantizer* inputQuantizer = new Quantizer(true);
	inputQuantizer->SetQuantizeParameters(Vector((float)1000.0), Vector((float)0.0), Vector((float)5000.0));

	// Initialize plant
	Rocket* plant = new Rocket();

	// Initialize controller
	Controller controller(plant, stateQuantizer, inputQuantizer);

	// Initialize a control specification
	ControlSpecification specification(ControlSpecificationType::Reachability, plant);
	specification.SetHyperInterval(Vector({ -1, -1 }), Vector({ 1, 1 }));
	controller.SetControlSpecification(&specification);

	// Initialize a multilayer perceptron neural network and configure it to function as controller
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 32, 32 }, ActivationActType::kRelu, LossFunctionType::CrossEntropy);
	multilayerPerceptron->InitializeOptimizer("adam", 0.0075, 0.001, false);
	multilayerPerceptron->ConfigurateInputOutput(plant, inputQuantizer, steps, 1.0);
	controller.SetNeuralNetwork(multilayerPerceptron);

	// Training routine for the neural network controller
	std::cout << "Training" << std::endl;
	for (int i = 0; i < episodes; i++) {
		if (i % verboseEpisode == 0) std::cout << std::endl;

		vector<Vector> states;
		vector<Vector> reinforcingLabels;
		vector<Vector> deterringLabels;

		// Get an initial state based on the control specification we are trying to solve for
		float progressionFactor = (float)i / (float)episodes;
		auto initialState = specification.GetCenter();

		switch (specification.GetSpecificationType()) {
		case ControlSpecificationType::Invariance:

			break;
		case ControlSpecificationType::Reachability:
			while (specification.IsInSpecificationSet(initialState)) {
				initialState = stateQuantizer->GetRandomVector();
				initialState[0] = initialState[0] * 0.1 + progressionFactor * 0.8;
				initialState[1] = initialState[1] * 0.2 + progressionFactor * 0.4;
			}
			break;
		}
		plant->SetState(initialState);

		// Define the norm for determining the networks performance
		vector<float> normWeights = { 1.0, 1.0 };
		auto initialNorm = (initialState - specification.GetCenter()).GetWeightedNorm(normWeights);
		auto oldNorm = initialNorm;

		bool isInSpecificationSet = false;

		// Simulate the episode using the neural network
		for (int j = 0; j < steps; j++) {
			auto normalizedQuantizedState = stateQuantizer->NormalizeVector(stateQuantizer->QuantizeVector(plant->GetState()));

			// Get network output for the quantized state
			auto networkOutput = multilayerPerceptron->EvaluateNetwork(normalizedQuantizedState);
			if(j == 0) networkOutput = multilayerPerceptron->EvaluateNetwork(normalizedQuantizedState); // TODO: Find a more expedient solution to this bug

			auto amountOfOutputNeurons = networkOutput.GetLength();

			// Get total probability of all the labels
			float totalProbability = 0.0;
			for (int i = 0; i < amountOfOutputNeurons; i++) totalProbability += networkOutput[i];

			// Normalize probabilities
			for (int i = 0; i < amountOfOutputNeurons; i++) networkOutput[i] = networkOutput[i] / totalProbability;

			// Determine a one hot vector from the network output
			float randomValue = ((float)rand() / RAND_MAX);
			float cumalitiveProbability = 0.0;
			int oneHotIndex = 0;
			for (int i = 0; i < amountOfOutputNeurons; i++) {
				cumalitiveProbability += networkOutput[i];
				if (randomValue < cumalitiveProbability) {
					oneHotIndex = i;
					break;
				}
			}

			// Determine input from one hot vector
			Vector oneHot(amountOfOutputNeurons);
			oneHot[oneHotIndex] = 1.0;

			auto input = inputQuantizer->FindVectorFromOneHot(oneHot);

			// Evolve the plant
			plant->Evolve(input);
			auto state = plant->GetState();

			// See if the evolved state is in the specification set and calculate the new norm on the state
			isInSpecificationSet = specification.IsInSpecificationSet(state);
			auto norm = (state - specification.GetCenter()).GetWeightedNorm(normWeights);

			// Find the reinforcing labels and the deterring labels
			Vector reinforcementLabel = Vector(networkOutput.GetLength());
			Vector deterringLabel = Vector(networkOutput.GetLength());

			reinforcementLabel = oneHot;

			float sum = 0.0;
			for (int i = 0; i < amountOfOutputNeurons; i++) {
				deterringLabel[i] = 1.0 - reinforcementLabel[i];
				sum += deterringLabel[i];
			}

			for (int i = 0; i < amountOfOutputNeurons; i++) deterringLabel[i] = deterringLabel[i] / sum;

			// Add states and labels to the list of states and labels
			states.push_back(normalizedQuantizedState);
			reinforcingLabels.push_back(reinforcementLabel);
			deterringLabels.push_back(deterringLabel);

			oldNorm = norm;

			// DEBUG: Print simulation for verification purposes
			if (i % verboseEpisode == 0) {
				auto verboseLabels = (amountOfOutputNeurons <= 5) ? amountOfOutputNeurons : 5;

				std::cout << "i: " << i << "\tj: " << j << "\t\tx0: " << state[0] << "\tx1: " << state[1];
				for (int i = 0; i < verboseLabels; i++) {
					if (i == 0) std::cout << "\t";
					std::cout << "\tp" << i << ": " << networkOutput[i];;
				}
				std::cout <<"\tu: " << input[0] << "\ts: " << isInSpecificationSet << std::endl;
			}

			// Check episode stopping conditions
			bool stopEpisode = false;
			switch (specification.GetSpecificationType()) {
			case ControlSpecificationType::Invariance:
				if (!isInSpecificationSet) stopEpisode = true;
				break;
			case ControlSpecificationType::Reachability:
				if (isInSpecificationSet) stopEpisode = true;
				break;
			}
			
			if (!stateQuantizer->IsInBounds(state) || stopEpisode) break;
		}

		// Train the neural network based on the performance of the network
		if (oldNorm < initialNorm || isInSpecificationSet) multilayerPerceptron->Train(states, reinforcingLabels);
		else multilayerPerceptron->Train(states, deterringLabels);
	}

	// Verification routine for the neural network controller


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