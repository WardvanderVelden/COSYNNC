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
	cosynnc.SpecifyStateQuantizer(Vector({ 0.005, 0.005 }), Vector({ 0.65, 4.95 }), Vector({ 1.65, 5.95 }));
	cosynnc.SpecifyInputQuantizer(Vector((float)0.5), Vector((float)0.0), Vector((float)1.0));

	// Specify the synthesis parameters
	cosynnc.SpecifySynthesisParameters(2500000, 50, 5000, 50000, 50);
	cosynnc.SpecifyRadialInitialState(0.25, 0.75);
	cosynnc.SpecifyNorm({ 1.0, 1.0 });
	cosynnc.SpecifyTrainingFocus(TrainingFocus::AlternatingRadialLosingNeighborLosing);

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 8, 8}, ActivationActType::kRelu, OutputType::Labelled);
	multilayerPerceptron->InitializeOptimizer("sgd", 0.01, 0.0); //0.001);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Invariance, Vector({ 0.7, 5.0 }), Vector({ 1.6, 5.9 }));

	// Specify how verbose the procedure should be
	cosynnc.SpecifyVerbosity(true, false);

	// Initialize the synthesize procedure
	cosynnc.Initialize();

	// Load a previously trained network
	//cosynnc.LoadNeuralNetwork("controllers/timestamps", "FriMar20115005net.m");

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
	cosynnc.SpecifyStateQuantizer(Vector({ 0.005, 0.005 }), Vector({ 0.65, 4.95 }), Vector({ 1.65, 5.95 })); // This is the setting as used by Rungger
	//cosynnc.SpecifyStateQuantizer(Vector({ 0.0075, 0.0075 }), Vector({ 0.65, 4.95 }), Vector({ 1.65, 5.95 }));
	cosynnc.SpecifyInputQuantizer(Vector((float)0.5), Vector((float)0.0), Vector((float)1.0));

	// Specify the synthesis parameters
	cosynnc.SpecifySynthesisParameters(1000000, 100, 5000, 50000, 50);
	cosynnc.SpecifyRadialInitialState(0.4, 0.6);
	cosynnc.SpecifyNorm({ 0.0, 1.0 });
	cosynnc.SpecifyWinningSetReinforcement(true);
	cosynnc.SpecifyTrainingFocus(TrainingFocus::AlternatingRadialLosingNeighborLosing);

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 8, 8 }, ActivationActType::kRelu, OutputType::Labelled);
	//multilayerPerceptron->InitializeOptimizer("sgd", 0.0025, 0.001);
	multilayerPerceptron->InitializeOptimizer("sgd", 0.01, 0.0);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Reachability, Vector({ 1.1, 5.4 }), Vector({ 1.6, 5.9 })); // Rungger specification

	// Specify how verbose the procedure should be
	cosynnc.SpecifyVerbosity(true, false);

	// Initialize the synthesize procedure
	cosynnc.Initialize();

	// Load a previously trained network
	//cosynnc.LoadNeuralNetwork("controllers/timestamps", "ThuMar26131918net.m"); 

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
	cosynnc.SpecifySynthesisParameters(2500000, 25, 5000, 50000, 50);
	cosynnc.SpecifyRadialInitialState(0.25, 0.75);
	cosynnc.SpecifyNorm({ 1.0, 1.0 });
	cosynnc.SpecifyTrainingFocus(TrainingFocus::RadialOutwards);

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 8, 8 }, ActivationActType::kRelu, OutputType::Labelled);
	multilayerPerceptron->InitializeOptimizer("sgd", 0.01, 0.0); // 01);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Invariance, Vector({ -2.5, -5.0 }), Vector({ 2.5, 5.0 }));

	// Specify how verbose the procedure should be
	cosynnc.SpecifyVerbosity(true, false);

	// Initialize the synthesize procedure
	cosynnc.Initialize();

	//cosynnc.LoadNeuralNetwork("controllers/timestamps","TueFeb25144915net.m");

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
	cosynnc.SpecifyStateQuantizer(Vector({ 0.1, 0.1 }), Vector({ -5, -10 }), Vector({ 5, 10 }));
	cosynnc.SpecifyInputQuantizer(Vector((float)1000.0), Vector((float)0.0), Vector((float)5000.0));

	// Specify the synthesis parameters
	cosynnc.SpecifySynthesisParameters(5000000, 50, 5000, 50000, 50);
	cosynnc.SpecifyRadialInitialState(0.15, 0.85);
	cosynnc.SpecifyNorm({ 1.0, 1.0 });
	cosynnc.SpecifyWinningSetReinforcement(true);
	cosynnc.SpecifyTrainingFocus(TrainingFocus::AlternatingRadialLosingNeighborLosing);

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 8, 8 }, ActivationActType::kRelu, OutputType::Labelled);
	//multilayerPerceptron->InitializeOptimizer("sgd", 0.005, 0.001);
	multilayerPerceptron->InitializeOptimizer("sgd", 0.01, 0.0);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Reachability, Vector({ -1.0, -1.0 }), Vector({ 1.0, 1.0 }));

	// Specify how verbose the procedure should be
	cosynnc.SpecifyVerbosity(true, false);

	// Initialize the synthesize procedure
	cosynnc.Initialize();

	// Load a previously trained network
	//cosynnc.LoadNeuralNetwork("controllers/timestamps", "TueMar24155649net.m");

	// Run the synthesize procedure
	cosynnc.Synthesize();

	// Free up memory
	delete rocket;
	delete multilayerPerceptron;
}


void LoadAndCompressController(string controllerName) {
	Procedure cosynnc;

	Plant* rocket = new Rocket();
	cosynnc.SetPlant(rocket);

	// Specify the state and input quantizers
	cosynnc.SpecifyStateQuantizer(Vector({ 0.1, 0.1 }), Vector({ -5, -10 }), Vector({ 5, 10 }));
	cosynnc.SpecifyInputQuantizer(Vector((float)250.0), Vector((float)0.0), Vector((float)5000.0));

	// Specify the synthesis parameters
	//cosynnc.SpecifySynthesisParameters(5000000, 50, 2500, 50000, 50);
	//cosynnc.SpecifyRadialInitialState(0.15, 0.85);
	//cosynnc.SpecifyNorm({ 1.0, 1.0 });
	//cosynnc.SpecifyTrainingFocus(TrainingFocus::AlternatingRadialLosingNeighborLosing);

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 8, 8 }, ActivationActType::kRelu, OutputType::Labelled);
	multilayerPerceptron->InitializeOptimizer("adam", 0.005, 0.001);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	//cosynnc.SpecifyControlSpecification(ControlSpecificationType::Reachability, Vector({ -1.0, -1.0 }), Vector({ 1.0, 1.0 }));

	// Specify how verbose the procedure should be
	//cosynnc.SpecifyVerbosity(true, false);

	// Load a previously trained network
	cosynnc.Initialize();
	cosynnc.LoadNeuralNetwork("controllers/timestamps", "ThuMar5104018net.m");
}


int main() {
	SynthesizeInvarianceControllerDCDC();
	//SynthesizeReachabilityControllerDCDC();
	
	//SynthesizeInvarianceControllerRocket();
	//SynthesizeReachabilityControllerRocket();

	system("pause");
	return 0;
}