#include <iostream>
#include "Vector.h"
#include "Plant.h"
#include "Quantizer.h"
#include "Controller.h"

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
	std::cout << "Hello COSYNNC!" << std::endl;

	// Initialize quantizer
	Quantizer* quantizer = new Quantizer();
	quantizer->SetStateQuantizeParameters({ 0.1, 0.1 }, { 0, 0 });
	quantizer->SetInputQuantizeParameters(vector<float>(1,0.1), vector<float>(1, 0));

	// Initialize plant
	Vector initialState({ 1, 1 });

	Rocket* plant = new Rocket();
	plant->SetState(initialState);

	// Initialize controller
	Controller controller(plant, quantizer);

	// Simulate closed loop
	std::cout << std::endl;
	for (int i = 0; i < 20; i++) {
		Vector input = controller.GetControlAction(quantizer->QuantizeToState(plant->GetState()));
		plant->Evolve(input);

		plant->GetState().PrintValues();
		std::cout << std::endl;
	}

	// Free memory
	delete plant;
	delete quantizer;
	
	return 0;
}