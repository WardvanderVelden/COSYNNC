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


	// Initialize quantizers
	Quantizer* stateQuantizer = new Quantizer(true);
	//stateQuantizer->SetQuantizeParameters(Vector({ 0.1, 0.1 }), Vector({ -0.05, -0.05 }));
	stateQuantizer->SetQuantizeParameters(Vector({ 0.1,0.1 }), Vector({ -10, -5 }), Vector({ 10, 5 }));

	Quantizer* inputQuantizer = new Quantizer(true);
	inputQuantizer->SetQuantizeParameters(Vector((float)25.0), Vector((float)0.0), Vector((float)5000.0));

	// Initialize plant
	Rocket* plant = new Rocket();
	plant->SetState(Vector({ 10, 0 }));


	// Initialize controller
	Controller controller(plant, stateQuantizer, inputQuantizer);


	// Initialize a control specification
	ControlSpecification specification(ControlSpecificationType::Reachability, plant);
	specification.SetHyperInterval(Vector({ -0.05, -0.05 }), Vector({ 0.05, 0.05 }));
	controller.SetControlSpecification(&specification);


	// Initialize a multilayer perceptron neural network
	// DEBUG: This is just a simple test 1 8 1 neural network to test the MXNet library functionality 
	// Please note that the output layer should be noted
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 128, 128, 64, 1 }, ActivationActType::kRelu);
	TrainingData* data = multilayerPerceptron->GetTrainingData(plant, &controller, stateQuantizer, inputQuantizer);
	multilayerPerceptron->Test(data, stateQuantizer, inputQuantizer);


	// Free memory
	delete data;
	delete plant;
	delete stateQuantizer;
	delete inputQuantizer;
	delete multilayerPerceptron;


	// Wait for an input to stop the program
	system("pause");
	return 0;
}