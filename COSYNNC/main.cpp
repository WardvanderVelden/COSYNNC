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

	// TEMPORARY: Example switch variables
	const bool isRocketExample = true;
	const bool isReachability = true;

	// COSYNNC training parameters
	const int episodes = 2500000;
	const int steps = 15;
	const int verboseEpisode = 2500;
	const int verificationEpisode = 25000;

	const bool verboseMode = true;

	// Initialize quantizers
	Quantizer* stateQuantizer = new Quantizer(true);
	if (isRocketExample)	stateQuantizer->SetQuantizeParameters(Vector({ 0.25, 0.5 }), Vector({ -5, -10 }), Vector({ 5, 10 }));
	//if (isRocketExample)	stateQuantizer->SetQuantizeParameters(Vector({ 0.125, 0.25 }), Vector({ -5, -10 }), Vector({ 5, 10 }));
	else stateQuantizer->SetQuantizeParameters(Vector({ 0.0125, 0.0125 }), Vector({ 1.15, 5.45 }), Vector({ 1.55, 5.85 }));

	Quantizer* inputQuantizer = new Quantizer(true);
	if(isRocketExample) inputQuantizer->SetQuantizeParameters(Vector((float)1000.0), Vector((float)0.0), Vector((float)5000.0));
	else inputQuantizer->SetQuantizeParameters(Vector((float)0.5), Vector((float)0.0), Vector((float)1.0));

	// Initialize plant
	Plant* plant;
	if(isRocketExample) plant = new Rocket();
	else plant = new DCDC();

	// Initialize controller
	Controller controller(plant, stateQuantizer, inputQuantizer);

	// Initialize a control specification
	ControlSpecification specification;
	if (isRocketExample) {
		if (isReachability) {
			specification = ControlSpecification(ControlSpecificationType::Reachability, plant);
			specification.SetHyperInterval(Vector({ -1, -1 }), Vector({ 1, 1 }));
			//specification.SetHyperInterval(Vector({ 0.5, -0.5 }), Vector({ 1.5, 0.5 }));
		}
		else {
			specification = ControlSpecification(ControlSpecificationType::Invariance, plant);
			specification.SetHyperInterval(Vector({ -2.5, -5 }), Vector({ 2.5, 5 }));
		}	
	}
	else {
		if (isReachability) {
			specification = ControlSpecification(ControlSpecificationType::Reachability, plant);
			specification.SetHyperInterval(Vector({ 1.35, 5.65 }), Vector({ 1.55, 5.85 }));
			//specification.SetHyperInterval(Vector({ 1.35, 5.65 }), Vector({ 1.45, 5.75 }));
		}
		else {
			specification = ControlSpecification(ControlSpecificationType::Invariance, plant);
			//specification.SetHyperInterval(Vector({ 1.15, 5.45 }), Vector({ 1.55, 5.85 }));
			specification.SetHyperInterval(Vector({ 1.175, 5.475 }), Vector({ 1.525, 5.825 }));
		}
	}
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
	for (int i = 0; i <= episodes; i++) {
		if (i % verboseEpisode == 0 && verboseMode) std::cout << std::endl;

		vector<Vector> states;
		vector<Vector> reinforcingLabels;
		vector<Vector> deterringLabels;

		// Get an initial state based on the control specification we are trying to solve for
		float progressionFactor = (float)i / (float)episodes;
		auto initialState = Vector({ 0.0, 0.0 }); 

		switch (specification.GetSpecificationType()) {
		case ControlSpecificationType::Invariance:
			initialState = specification.GetVectorFromSpecification();
			break;
		case ControlSpecificationType::Reachability:
			//initialState = verifier->GetVectorFromLosingDomain();
			initialState = verifier->GetVectorRadialFromGoal(0.4 + 0.6 * progressionFactor);
			
			while (specification.IsInSpecificationSet(initialState)) {
				initialState = verifier->GetVectorRadialFromGoal(0.4 + 0.6 * progressionFactor);
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
			if (i % verboseEpisode == 0 && verboseMode) {
				auto verboseLabels = (inputQuantizer->GetCardinality() <= 5) ? inputQuantizer->GetCardinality() : 5;

				std::cout << "i: " << i << "\tj: " << j << "\t\tx0: " << newState[0] << "\tx1: " << newState[1];
				for (int i = 0; i < verboseLabels; i++) {
					if (i == 0) std::cout << "\t";
					std::cout << "\tn" << i << ": " << networkOutput[i];;
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

		//if (isInSpecificationSet) multilayerPerceptron->Train(states, reinforcingLabels);
		//else multilayerPerceptron->Train(states, deterringLabels);

		// Verification routine for the neural network controller
		if (i % verificationEpisode == 0 && i != 0) {
			std::cout << std::endl;

			verifier->ComputeTransitionFunction();
			verifier->ComputeWinningSet();

			auto winningSetSize = verifier->GetWinningSetSize();
			float winningSetPercentage = (float)winningSetSize / (float)stateQuantizer->GetCardinality();

			// Empirical random walks
			if (verboseMode) {
				std::cout << "Empirical verification" << std::endl;
				for (unsigned int j = 0; j < 5; j++) {
					std::cout << std::endl;
					auto initialState = stateQuantizer->GetRandomVector();
					verifier->PrintVerboseWalk(initialState);
				}
				std:cout << std::endl;
			}

			// Winning set
			std::cout << "Winning set size percentage: " << winningSetPercentage * 100 << "%" << std::endl << std::endl;
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
	std::cout << "COSYNNC: Epoch limit reached" << std::endl;

	system("pause");
	return 0;
}