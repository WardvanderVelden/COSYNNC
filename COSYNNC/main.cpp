#include <iostream>
#include "Vector.h"
#include "Plant.h"
#include "Rocket.h"
#include "DCDC.h"
#include "Quantizer.h"
#include "Controller.h"
#include "ControlSpecification.h"
#include "MultilayerPerceptron.h"
#include "Verifier.h"

using namespace std;
using namespace mxnet;

int main() {
	std::cout << "COSYNNC: A correct-by-design neural network synthesis tool." << std::endl << std::endl;

	// COSYNNC training parameters
	const int episodes = 1000000;
	const int steps = 25; // 25;
	const int verboseEpisode = 2500;
	const int verificationEpisode = 50000;

	// Initialize quantizers
	Quantizer* stateQuantizer = new Quantizer(true);
	stateQuantizer->SetQuantizeParameters(Vector({ 0.1, 0.1 }), Vector({ -5, -10 }), Vector({ 5, 10 }));
	//stateQuantizer->SetQuantizeParameters(Vector({ 0.0125, 0.0125 }), Vector({ 1.15, 5.45 }), Vector({ 1.55, 5.85 }));

	Quantizer* inputQuantizer = new Quantizer(true);
	inputQuantizer->SetQuantizeParameters(Vector((float)1000.0), Vector((float)0.0), Vector((float)5000.0));
	//inputQuantizer->SetQuantizeParameters(Vector((float)0.5), Vector((float)0.0), Vector((float)1.0));

	// Initialize plant
	Rocket* plant = new Rocket();
	//DCDC* plant = new DCDC();

	// Initialize controller
	Controller controller(plant, stateQuantizer, inputQuantizer);

	// Initialize a control specification
	ControlSpecification specification(ControlSpecificationType::Reachability, plant);
	specification.SetHyperInterval(Vector({ -1, -1 }), Vector({ 1, 1 }));
	//ControlSpecification specification(ControlSpecificationType::Invariance, plant);
	//specification.SetHyperInterval(Vector({ 1.15, 5.45 }), Vector({ 1.55, 5.85 }));
	//ControlSpecification specification(ControlSpecificationType::Reachability, plant);
	//specification.SetHyperInterval(Vector({ 1.45, 5.45 }), Vector({ 1.55, 5.85 }));
	controller.SetControlSpecification(&specification);

	// Initialize a multilayer perceptron neural network and configure it to function as controller
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 32, 32 }, ActivationActType::kRelu, LossFunctionType::CrossEntropy);
	multilayerPerceptron->InitializeOptimizer("adam", 0.0075, 0.001, false);
	multilayerPerceptron->ConfigurateInputOutput(plant, inputQuantizer, steps, 1.0);
	controller.SetNeuralNetwork(multilayerPerceptron);

	// Initialize the verifier for verifying the controller
	Verifier* verifier = new Verifier(plant, &controller, stateQuantizer, inputQuantizer);

	// Training routine for the neural network controller
	std::cout << "Training" << std::endl;
	for (int i = 0; i < episodes; i++) {
		if (i % verboseEpisode == 0) std::cout << std::endl;

		vector<Vector> states;
		vector<Vector> reinforcingLabels;
		vector<Vector> deterringLabels;

		// Get an initial state based on the control specification we are trying to solve for
		float progressionFactor = (float)i / (float)episodes;
		auto initialState = Vector({ 0.0, 0.0 }); 
		//auto initialState = Vector({ 1.2, 5.6 });
		//auto initialState = stateQuantizer->GetRandomVector();

		switch (specification.GetSpecificationType()) {
		case ControlSpecificationType::Invariance:
			break;
		case ControlSpecificationType::Reachability:
			initialState = verifier->GetVectorRadialFromGoal(0.15 + 0.85 * progressionFactor);
			//initialState = verifier->GetVectorFromLosingDomain();
			while (specification.IsInSpecificationSet(initialState)) {
				initialState = verifier->GetVectorRadialFromGoal(0.15 + 0.85 * progressionFactor);
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
			auto state = plant->GetState();
			auto normalizedQuantizedState = stateQuantizer->NormalizeVector(stateQuantizer->QuantizeVector(state));

			Vector oneHot(inputQuantizer->GetCardinality());
			Vector networkOutput(inputQuantizer->GetCardinality());
			auto input = controller.GetProbabilisticControlAction(state, oneHot, networkOutput);

			// Evolve the plant
			plant->Evolve(input);
			auto newState = plant->GetState();

			// See if the evolved state is in the specification set and calculate the new norm on the state
			isInSpecificationSet = specification.IsInSpecificationSet(newState);
			auto norm = (newState - specification.GetCenter()).GetWeightedNorm(normWeights);

			// Find the reinforcing labels and the deterring labels
			Vector reinforcementLabel = Vector(inputQuantizer->GetCardinality());
			Vector deterringLabel = Vector(inputQuantizer->GetCardinality());

			reinforcementLabel = oneHot;

			float sum = 0.0;
			for (int i = 0; i < inputQuantizer->GetCardinality(); i++) {
				deterringLabel[i] = 1.0 - reinforcementLabel[i];
				sum += deterringLabel[i];
			}

			for (int i = 0; i < inputQuantizer->GetCardinality(); i++) deterringLabel[i] = deterringLabel[i] / sum;

			// Add states and labels to the list of states and labels
			states.push_back(normalizedQuantizedState);
			reinforcingLabels.push_back(reinforcementLabel);
			deterringLabels.push_back(deterringLabel);

			oldNorm = norm;

			// DEBUG: Print simulation for verification purposes
			if (i % verboseEpisode == 0) {
				auto verboseLabels = (inputQuantizer->GetCardinality() <= 5) ? inputQuantizer->GetCardinality() : 5;

				std::cout << "i: " << i << "\tj: " << j << "\t\tx0: " << newState[0] << "\tx1: " << newState[1];
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
			
			if (!stateQuantizer->IsInBounds(newState) || stopEpisode) break;
		}

		// Train the neural network based on the performance of the network
		if (oldNorm < initialNorm || isInSpecificationSet) multilayerPerceptron->Train(states, reinforcingLabels);
		else multilayerPerceptron->Train(states, deterringLabels);

		// Verification routine for the neural network controller
		if (i % verificationEpisode == 0 && i != 0) {
			verifier->ComputeTransitionFunction();
			verifier->ComputeWinningSet();

			auto winningSetSize = verifier->GetWinningSetSize();
			float winningSetPercentage = (float)winningSetSize / (float)stateQuantizer->GetCardinality();

			std::cout << std::endl << "Winning set size percentage: " << winningSetPercentage * 100 << "%" << std::endl;
		}
	}

	// Free memory
	delete plant;
	delete stateQuantizer;
	delete inputQuantizer;
	delete multilayerPerceptron;
	delete verifier;

	MXNotifyShutdown();

	// Wait for an input to stop the program
	system("pause");
	return 0;
}