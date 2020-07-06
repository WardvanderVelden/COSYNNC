#include <iostream>

#include "Vector.h"
#include "Plant.h"
#include "Rocket.h"
#include "DCDC.h"
#include "StateSpaceRepresentation.h"
#include "Unicycle.h"

#include "Quantizer.h"
#include "Controller.h"
#include "ControlSpecification.h"
#include "MultilayerPerceptron.h"
#include "Verifier.h"
#include "Procedure.h"
#include "Encoder.h"

using namespace std;
using namespace mxnet;


void SynthesizeReachabilityControllerRocket() {
	Procedure cosynnc;

	// Link the plant to the procedure
	Plant* rocket = new Rocket();
	cosynnc.SetPlant(rocket);

	// Specify the state and input quantizers
	cosynnc.SpecifyStateQuantizer(Vector({ 0.1, 0.1 }), Vector({ -5, -10 }), Vector({ 5, 10 }));
	cosynnc.SpecifyInputQuantizer(Vector((float)1000.0), Vector((float)0.0), Vector((float)5000.0));

	// Specify the synthesis parameters
	cosynnc.SpecifySynthesisParameters(1000000, 50, 5000, 50000, 50);

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 8, 8 }, ActivationActType::kRelu, OutputType::Labelled);
	multilayerPerceptron->InitializeOptimizer("sgd", 0.0075, 0.0);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Reachability, Vector({ -1.0, -1.0 }), Vector({ 1.0, 1.0 }));

	cosynnc.SpecifyRadialInitialState(0.15, 0.85);
	cosynnc.SpecifyTrainingFocus(TrainingFocus::RadialOutwards);

	cosynnc.SpecifyTrainingFocus(TrainingFocus::LosingStates);

	cosynnc.SpecifyTrainingFocus(TrainingFocus::AllStates);

	//cosynnc.SpecifyNorm({ 1.0, 1.0 });
	cosynnc.SpecifyWinningSetReinforcement(true);

	cosynnc.SpecifyUseRefinedTransitions(true);

	cosynnc.SpecifySavingPath("../controllers");

	// Initialize the synthesize procedure
	cosynnc.Initialize();

	// Load a previously trained network
	//cosynnc.LoadNeuralNetwork("../controllers/timestamps", "net");

	// Run the synthesize procedure
	cosynnc.Synthesize();

	// Free up memory
	delete rocket;
	delete multilayerPerceptron;
}


void EncodeWinningSetAsNeuralNetwork() {
	Encoder encoder("../controllers/timestamps", "scs"); // This should be an existing static controller

	MultilayerPerceptron* mlp = new MultilayerPerceptron({ 8, 8 }, ActivationActType::kRelu, LossFunctionType::CrossEntropy);
	mlp->InitializeOptimizer("sgd", 0.0075, 0.0);

	encoder.SetBatchSize(10);
	encoder.SetNeuralNetwork(mlp);

	encoder.SetSavingPath("../controllers");
	
	encoder.Encode(10, 2, 97.9, 0.00499999);

	delete mlp;
}


int main() {
	SynthesizeReachabilityControllerRocket();

	//EncodeWinningSetAsNeuralNetwork();

	system("pause");
	return 0;
}