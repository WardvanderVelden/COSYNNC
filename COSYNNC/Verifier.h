#pragma once
#include "Plant.h"
#include "Controller.h"
#include "Quantizer.h"

namespace COSYNNC {
	class Verifier {
	public:
		Verifier(Plant* plant, Controller* controller, Quantizer* stateQuantizer, Quantizer* inputQuantizer);

		~Verifier();

		// Computes the transition function that transitions any state in the state space to a set of states in the state space based on the control law
		void ComputeTransitionFunction();

		// Computes the winning set for which the controller currently is able to adhere to the control specification
		void ComputeWinningSet();

		//  Prints a verbose walk of the current controller using greedy inputs
		void PrintVerboseWalk(Vector initialState);

		// Returns the size of the winning set compared to the cardinality of the state space
		long GetWinningSetSize();

		// Sets a part of the winning domain, returns whether or not that element has changed
		bool SetWinningDomain(long index, bool value);

		// Get a random vector from the space of the lossing domain
		Vector GetVectorFromLosingDomain();

		// Get a random vector in a radius to the goal based on training time
		Vector GetVectorRadialFromGoal(float progression);
	private:
		Plant* _plant;
		Controller* _controller;
		Quantizer* _stateQuantizer;
		Quantizer* _inputQuantizer;
		ControlSpecification* _specification;

		long* _transitions;
		bool* _winningSet;
		vector<long> _losingIndices;

		const unsigned int _maxSteps = 50;
	};
}