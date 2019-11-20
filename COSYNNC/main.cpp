#include <iostream>
#include "Vector.h"
#include "Plant.h"
#include "Quantizer.h"
#include "Controller.h"
#include "ControlSpecification.h"

using namespace std;

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
	std::cout << "COSYNNC: A correct-by-design neural network synthesis tool." << std::endl;


	// Initialize quantizers
	Quantizer* stateQuantizer = new Quantizer();
	stateQuantizer->SetQuantizeParameters(Vector({ 0.1, 0.1 }), Vector({ -0.05, -0.05 }));

	Quantizer* inputQuantizer = new Quantizer();
	inputQuantizer->SetQuantizeParameters(vector<float>{0.1}, vector<float>{-0.05});


	// Initialize plant
	Rocket* plant = new Rocket();
	plant->SetState(Vector({ 20, 0 }));


	// Initialize controller
	Controller controller(plant, stateQuantizer, inputQuantizer);


	// Initialize a control specification
	ControlSpecification specification(ControlSpecificationType::Reachability, plant);
	specification.SetHyperInterval(Vector({ -0.05, -0.05 }), Vector({ 0.05, 0.05 }));
	controller.SetControlSpecification(&specification);


	// Simulate closed loop
	std::cout << std::endl;
	for (int i = 0; i < 100; i++) {
		// Get the control action
		Vector input = controller.GetControlAction(plant->GetState());
		std::cout << "i: ";
		input.PrintValues();

		// Evolve the system according to the control action
		plant->Evolve(input);

		// Print state and quantized state for comparison
		std::cout << "\ts: ";
		plant->GetState().PrintValues();

		std::cout << "\tq: ";
		stateQuantizer->QuantizeVector(plant->GetState()).PrintValues();

		std::cout << std::endl;
	}
	std::cout << std::endl;


	// Free memory
	delete plant;
	delete stateQuantizer;
	delete inputQuantizer;


	// Wait for an input to stop the program
	system("pause");
	return 0;
}