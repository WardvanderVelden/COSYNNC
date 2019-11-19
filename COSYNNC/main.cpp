#include <iostream>
#include "Vector.h"
#include "Plant.h"

using namespace std;

class Rocket : public Plant {
public:
	Rocket() : Plant(2, 1, 0.1) { }

	// Simple second order rocket dynamics in one axis
	Vector SingleStepDynamics(Vector input) override {
		Vector newState(GetStateDimension());

		newState[0] = GetState()[0] + GetState()[1] * GetTau();
		newState[1] = GetState()[1] + GetTau() / _mass * input[0] + _g * GetTau();

		return newState;
	}
private:
	float _mass = 267; // kg
	float _g = -9.81; // m s^-2
}; 

int main() {
	std::cout << "Hello COSYNNC!" << std::endl << std::endl;

	Vector input(1);
	input[0] = 1000;

	Rocket plant;


	
	return 0;
}