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
	//stateQuantizer->SetQuantizeParameters(Vector({ 0.1,0.1 }), Vector({ -10, -5 }), Vector({ 10, 5 }));
	stateQuantizer->SetQuantizeParameters(Vector({ 0.1, 0.1 }), Vector({ -2, -1 }), Vector({ 2, 1 }));

	Quantizer* inputQuantizer = new Quantizer(true);
	inputQuantizer->SetQuantizeParameters(Vector((float)100.0), Vector((float)0.0), Vector((float)3000.0));

	// Initialize plant
	Rocket* plant = new Rocket();
	plant->SetState(Vector({ 0, 0 }));


	// Initialize controller
	Controller controller(plant, stateQuantizer, inputQuantizer);


	// Initialize a control specification
	ControlSpecification specification(ControlSpecificationType::Reachability, plant);
	specification.SetHyperInterval(Vector({ -0.05, -0.05 }), Vector({ 0.05, 0.05 }));
	controller.SetControlSpecification(&specification);


	// Initialize a multilayer perceptron neural network
	MultilayerPerceptron* multilayerPerceptron = new MultilayerPerceptron({ 16, 16, 1 }, ActivationActType::kRelu);
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