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


void SynthesizeReachabilityControllerDCDC() {
	Procedure cosynnc;
	 
	// Link the plant to the procedure
	Plant* dcdc = new DCDC();
	cosynnc.SetPlant(dcdc);

	// Specify the state and input quantizers
	cosynnc.SpecifyStateQuantizer(Vector({ 0.005, 0.005 }), Vector({ 0.65, 4.95 }), Vector({ 1.65, 5.95 })); // This is the setting as used by Rungger
	cosynnc.SpecifyInputQuantizer(Vector((float)1.0), Vector((float)0.0), Vector((float)1.0));

	// Specify the synthesis parameters
	cosynnc.SpecifySynthesisParameters(1000000, 50, 5000, 50000, 50);

	// Link a neural network to the procedure
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 10, 10 }, ActivationActType::kRelu, OutputType::Labelled);
	multilayerPerceptron->InitializeOptimizer("sgd", 0.0075, 0.0);
	cosynnc.SetNeuralNetwork(multilayerPerceptron);

	// Specify the control specification
	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Reachability, Vector({ 1.1, 5.4 }), Vector({ 1.6, 5.9 })); // Rungger specification
	//cosynnc.SpecifyControlSpecification(ControlSpecificationType::ReachAndStay, Vector({ 1.1, 5.4 }), Vector({ 1.6, 5.9 }));

	cosynnc.SpecifyRadialInitialState(0.4, 0.6);

	cosynnc.SpecifyTrainingFocus(TrainingFocus::RadialOutwards);
	cosynnc.SpecifyTrainingFocus(TrainingFocus::AllStates);

	cosynnc.SpecifyWinningSetReinforcement(true);

	cosynnc.SpecifyUseRefinedTransitions(true);

	cosynnc.SpecifyVerbosity(true, false);

	cosynnc.SpecifySavingPath("../controllers");

	// Initialize the synthesize procedure
	cosynnc.Initialize();

	// Load a previously trained network
	//cosynnc.LoadNeuralNetwork("controllers/timestamps", "WedMay20155901net"); 

	// Run the synthesize procedure
	cosynnc.Synthesize();
	
	// Free up memory
	delete dcdc;
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
	//cosynnc.LoadNeuralNetwork("controllers/timestamps", "ThuApr30113252net");

	// Run the synthesize procedure
	cosynnc.Synthesize();

	// Free up memory
	delete rocket;
	delete multilayerPerceptron;
}


void SynthesizeSS3dReachabilityController() {
	// Define plant
	StateSpaceRepresentation* plant = new StateSpaceRepresentation(3, 1, 0.05, "Linear 3");
	double** A = new double* [3];
	A[0] = new double[3] { 1.6128, 1.9309, 2.2273 };
	A[1] = new double[3] { 1.3808, 0.5701, 2.1409};
	A[2] = new double[3] { 0.5453, 0.9272, 1.0061};

	double** B = new double* [3];
	B[0] = new double[1] {0.7950};
	B[1] = new double[1] {1.5216};
	B[2] = new double[1] {2.2755};

	plant->SetMatrices(A, B);

	// Delete 2d matrix
	for (size_t i = 0; i < 3; i++) {
		delete[] A[i];
		delete[] B[i];
	}
	delete[] A;
	delete[] B;

	// Define network
	MultilayerPerceptron* mlp = new MultilayerPerceptron({ 8, 8 }, ActivationActType::kRelu, OutputType::Labelled);
	mlp->InitializeOptimizer("sgd", 0.0075, 0.0);

	Procedure cosynnc;
	cosynnc.SetPlant(plant);

	cosynnc.SpecifyStateQuantizer(Vector({ 0.25, 0.25, 0.25}), Vector({ -5.0, -5.0, -5.0 }), Vector({ 5.0, 5.0, 5.0 }));
	cosynnc.SpecifyInputQuantizer(Vector({ 2.5 }), Vector({ -5.0 }), Vector({ 5.0 }));

	cosynnc.SpecifySynthesisParameters(5000000, 50, 5000, 50000, 50);

	cosynnc.SetNeuralNetwork(mlp, 5);

	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Reachability, Vector({ -1.0, -1.0, -1.0 }), Vector({ 1.0, 1.0, 1.0 }));

	cosynnc.SpecifyWinningSetReinforcement(true);
	cosynnc.SpecifyTrainingFocus(TrainingFocus::NeighboringLosingStates);

	cosynnc.SpecifyTrainingFocus(TrainingFocus::AllStates);

	cosynnc.SpecifyVerbosity(true, false);

	cosynnc.SpecifyUseRefinedTransitions(false);

	cosynnc.SpecifySavingPath("../controllers");

	cosynnc.Initialize();

	cosynnc.Synthesize();

	delete mlp;
	delete plant;
}


void SynthesizeMIMOReachabilityController() {
	// Define plant
	StateSpaceRepresentation* plant = new StateSpaceRepresentation(2, 2, 0.025, "Linear 2");
	double** A = new double* [2];
	A[0] = new double[2]{ 3.8045, 0.7585 };
	A[1] = new double[2]{ 5.6782, 0.5395 };

	double** B = new double* [2];
	B[0] = new double[2]{ 5.3080, 9.3401 };
	B[1] = new double[2]{ 7.7917, 1.2991 };

	plant->SetMatrices(A, B);

	// Delete 2d matrix
	for (size_t i = 0; i < 2; i++) {
		delete[] A[i];
		delete[] B[i];
	}
	delete[] A;
	delete[] B;

	// Define network
	MultilayerPerceptron* mlp = new MultilayerPerceptron({ 8, 8 }, ActivationActType::kRelu, OutputType::Labelled);
	mlp->InitializeOptimizer("sgd", 0.0075, 0.0);

	Procedure cosynnc;
	cosynnc.SetPlant(plant);

	cosynnc.SpecifyStateQuantizer(Vector({ 0.1, 0.1 }), Vector({ -5.0, -5.0 }), Vector({ 5.0, 5.0 }));
	cosynnc.SpecifyInputQuantizer(Vector({ 5.0, 2.0 }), Vector({ -5.0, -2.0 }), Vector({ 5.0, 2.0 }));

	cosynnc.SpecifySynthesisParameters(2000000, 25, 5000, 50000, 50);
	cosynnc.SpecifyWinningSetReinforcement(true);

	cosynnc.SpecifyRadialInitialState(0.0, 0.9);
	cosynnc.SpecifyTrainingFocus(TrainingFocus::RadialOutwards);
	cosynnc.SpecifyTrainingFocus(TrainingFocus::NeighboringLosingStates);

	cosynnc.SetNeuralNetwork(mlp, 9);

	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Invariance, Vector({ -2.505, -2.505 }), Vector({ 2.505, 2.505 }));

	cosynnc.SpecifyVerbosity(true, false);
	cosynnc.SpecifyUseRefinedTransitions(true);
	cosynnc.SpecifyComputeApparentWinningSet(true);

	cosynnc.SpecifySavingPath("../controllers");

	cosynnc.Initialize();

	//cosynnc.LoadNeuralNetwork("controllers/timestamps", "MonMay11151654net");

	cosynnc.Synthesize();

	delete mlp;
	delete plant;
}


void SynthesizeUnicycleReachabilityController() {
	// Define plant
	Unicycle* plant = new Unicycle();

	// Define network
	MultilayerPerceptron* mlp = new MultilayerPerceptron({ 12, 12 }, ActivationActType::kRelu, OutputType::Labelled);
	mlp->InitializeOptimizer("sgd", 0.0075, 0.0);

	Procedure cosynnc;
	cosynnc.SetPlant(plant);

	//cosynnc.SpecifyStateQuantizer(Vector({ 0.2, 0.2, PI/6 }), Vector({ -5.0, -5.0, 0 }), Vector({ 5.0, 5.0, 2*PI }));
	cosynnc.SpecifyStateQuantizer(Vector({ 0.1, 0.1, 0.1 }), Vector({ -5.0, -5.0, 0 }), Vector({ 5.0, 5.0, 2 * PI }));
	cosynnc.SpecifyInputQuantizer(Vector({ 1.0 }), Vector({ -1.0 }), Vector({ 1.0 }));

	cosynnc.SpecifySynthesisParameters(5000000, 50, 5000, 50000, 100);

	cosynnc.SetNeuralNetwork(mlp, 5);

	cosynnc.SpecifyControlSpecification(ControlSpecificationType::Reachability, Vector({ -2.05, -2.05, 0 }), Vector({ 2.05, 2.05, 2*PI }));

	cosynnc.SpecifyNorm({ 1.0, 1.0, 0.0 });
	cosynnc.SpecifyWinningSetReinforcement(true);

	//cosynnc.SpecifyTrainingFocus(TrainingFocus::RadialOutwards);
	//cosynnc.SpecifyRadialInitialState(0.5, 1.0);

	//cosynnc.SpecifyTrainingFocus(TrainingFocus::NeighboringLosingStates);

	cosynnc.SpecifyTrainingFocus(TrainingFocus::AllStates);

	cosynnc.SpecifyUseRefinedTransitions(true);

	cosynnc.SpecifySavingPath("../controllers");

	cosynnc.Initialize();

	//cosynnc.LoadNeuralNetwork("controllers/timestamps", "TueJun2102314net");

	cosynnc.Synthesize();

	delete mlp;
	delete plant;
}


void EncodeWinningSetAsNeuralNetwork() {
	Encoder encoder("../controllers/timestamps", "MonJun22115227scs"); // This should be an existing static controller

	MultilayerPerceptron* mlp = new MultilayerPerceptron({ 8, 8 }, ActivationActType::kRelu, LossFunctionType::CrossEntropy);
	mlp->InitializeOptimizer("sgd", 0.0075, 0.0);

	encoder.SetBatchSize(10);
	encoder.SetNeuralNetwork(mlp);

	encoder.SetSavingPath("../controllers");
	
	encoder.Encode(10, 2, 97.9, 0.00499999);

	delete mlp;
}


int main() {
	//SynthesizeReachabilityControllerDCDC();
	
	SynthesizeReachabilityControllerRocket();

	//SynthesizeSS3dReachabilityController();

	//SynthesizeMIMOReachabilityController();

	//SynthesizeUnicycleReachabilityController();

	//EncodeWinningSetAsNeuralNetwork();

	system("pause");
	return 0;
}