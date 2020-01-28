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
#include "Procedure.h"

using namespace std;
using namespace mxnet;


void SynthesizeInvarianceControllerDCDC() {
	Procedure cosynnc;

	// Link the plant to the procedure
	Plant* dcdc = new DCDC();
	cosynnc.SetPlant(dcdc);

	// Specify the state and input quantizers
	cosynnc.SpecifyStateQuantizer(Vector({ 0.01, 0.01 }), Vector({ 1.15, 5.45 }), Vector({ 1.55, 5.85 }));
	cosynnc.SpecifyInputQuantizer(Vector((float)0.5), Vector((float)0.0), Vector((float)1.0));

	// Specify the synthesis parameters
	cosynnc.SpecifySynthesisParameters(1000000, 25, 2500, 25000, 50);
	cosynnc.SpecifyRadialInitialState(0.25, 0.25);
	cosynnc.SpecifyNorm({ 1.0, 1.0 });

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 16, 16 }, ActivationActType::kRelu, LossFunctionType::CrossEntropy, OutputType::Labelled);
	multilayerPerceptron->InitializeOptimizer("adam", 0.005, 0.001, false);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Invariance, Vector({ 1.2, 5.5 }), Vector({ 1.5, 5.8 }));

	// Run the synthesize procedure
	cosynnc.Synthesize();

	// Free up memory
	delete dcdc;
	delete multilayerPerceptron;
}


void SynthesizeReachabilityControllerDCDC() {
	Procedure cosynnc;

	// Link the plant to the procedure
	Plant* dcdc = new DCDC();
	cosynnc.SetPlant(dcdc);

	// Specify the state and input quantizers
	//cosynnc.SpecifyStateQuantizer(Vector({ 0.005, 0.005 }), Vector({ 1.15, 5.45 }), Vector({ 1.55, 5.85 }));
	cosynnc.SpecifyStateQuantizer(Vector({ 0.005, 0.005 }), Vector({ 0.65, 4.95 }), Vector({ 1.65, 5.95 })); // This is the setting as used by Rungger
	//cosynnc.SpecifyStateQuantizer(Vector({ 0.01, 0.01 }), Vector({ 0.65, 4.95 }), Vector({ 1.65, 5.95 }));
	cosynnc.SpecifyInputQuantizer(Vector((float)0.5), Vector((float)0.0), Vector((float)1.0));

	// Specify the synthesis parameters
	cosynnc.SpecifySynthesisParameters(1000000, 10, 2500, 50000, 50);
	//cosynnc.SpecifyRadialInitialState(0.25, 0.25);
	cosynnc.SpecifyRadialInitialState(0.5, 0.8);
	cosynnc.SpecifyNorm({ 1.0, 1.0 });

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 8, 8 }, ActivationActType::kRelu, LossFunctionType::CrossEntropy, OutputType::Labelled);
	multilayerPerceptron->InitializeOptimizer("adam", 0.005, 0.001, false);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	//cosynnc.SpecifyControlSpecification(ControlSpecificationType::Reachability, Vector({ 1.35, 5.65 }), Vector({ 1.45, 5.75 }));
	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Reachability, Vector({ 1.1, 5.4 }), Vector({ 1.6, 5.9 }));

	// Specify how verbose the procedure should be
	cosynnc.SpecifyVerbosity(true, false);

	// Run the synthesize procedure
	cosynnc.Synthesize();

	// Free up memory
	delete dcdc;
	delete multilayerPerceptron;
}


void SynthesizeInvarianceControllerRocket() {
	Procedure cosynnc;

	// Link the plant to the procedure
	Plant* rocket = new Rocket();
	cosynnc.SetPlant(rocket);

	// Specify the state and input quantizers
	cosynnc.SpecifyStateQuantizer(Vector({ 0.1, 0.1 }), Vector({ -5, -10 }), Vector({ 5, 10 }));
	cosynnc.SpecifyInputQuantizer(Vector((float)500.0), Vector((float)0.0), Vector((float)5000.0));

	// Specify the synthesis parameters
	cosynnc.SpecifySynthesisParameters(1000000, 10, 2500, 25000, 50);
	cosynnc.SpecifyRadialInitialState(0.25, 0.75);
	cosynnc.SpecifyNorm({ 1.0, 2.0 });

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 8, 8, 8, 8 }, ActivationActType::kRelu, LossFunctionType::CrossEntropy, OutputType::Labelled);
	multilayerPerceptron->InitializeOptimizer("adam", 0.005, 0.001, false);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Invariance, Vector({ -4.0, -9.0 }), Vector({ 4, 9.0 }));

	// Run the synthesize procedure
	cosynnc.Synthesize();

	// Free up memory
	delete rocket;
	delete multilayerPerceptron;
}


void SynthesizeReachabilityControllerRocket() {
	Procedure cosynnc;

	// Link the plant to the procedure
	Plant* rocket = new Rocket();
	cosynnc.SetPlant(rocket);

	// Specify the state and input quantizers
	//cosynnc.SpecifyStateQuantizer(Vector({ 0.25, 0.5 }), Vector({ -5, -10 }), Vector({ 5, 10 }));
	cosynnc.SpecifyStateQuantizer(Vector({ 0.05, 0.1 }), Vector({ -5, -10 }), Vector({ 5, 10 }));
	cosynnc.SpecifyInputQuantizer(Vector((float)500.0), Vector((float)0.0), Vector((float)5000.0));

	// Specify the synthesis parameters
	cosynnc.SpecifySynthesisParameters(1000000, 10, 2500, 25000, 50);
	cosynnc.SpecifyRadialInitialState(0.25, 0.75);
	cosynnc.SpecifyNorm({ 1.0, 2.0 });

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 16, 16 }, ActivationActType::kRelu, LossFunctionType::CrossEntropy, OutputType::Labelled);
	multilayerPerceptron->InitializeOptimizer("adam", 0.005, 0.001, false);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Reachability, Vector({ -1.0, -1.0 }), Vector({ 1.0, 1.0 }));

	// Specify how verbose the procedure should be
	cosynnc.SpecifyVerbosity(true, false);

	// Run the synthesize procedure
	cosynnc.Synthesize();

	// Free up memory
	delete rocket;
	delete multilayerPerceptron;
}


int main() {
	//SynthesizeInvarianceControllerDCDC();
	SynthesizeReachabilityControllerDCDC();
	
	//SynthesizeInvarianceControllerRocket();
	//SynthesizeReachabilityControllerRocket();

	system("pause");
	return 0;
}