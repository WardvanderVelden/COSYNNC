#include "Verifier.h"

namespace COSYNNC {
	Verifier::Verifier(Plant* plant, Controller* controller, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		_plant = plant;
		_controller = controller;
		_stateQuantizer = stateQuantizer;
		_inputQuantizer = inputQuantizer;

		_specification = controller->GetControlSpecification();
	}


	Verifier::~Verifier() {
		delete[] _transitions;
		delete[] _winningSet;

		//for (long index = 0; index < _stateQuantizer->GetCardinality(); index++) delete _transitions[index];
	}


	// Calculates the transition function that transitions any state in the state space to a set of states in the state space based on the control law
	void Verifier::ComputeTransitionFunction() {
		auto stateSpaceCardinality = _stateQuantizer->GetCardinality();

		delete[] _transitions;
		_transitions = new long[stateSpaceCardinality];

		for (long index = 0; index < stateSpaceCardinality; index++) {
			// Get state based on the index
			auto state = _stateQuantizer->GetVectorFromIndex(index);
			_plant->SetState(state);

			// Find the input based on the current controller
			auto input = _controller->GetControlAction(state);
			
			// Calculate the transition
			_plant->Evolve(input);
			auto newState = _plant->GetState();

			long newIndex = -1;
			if (_stateQuantizer->IsInBounds(newState)) {
				auto quantizedNewState = _stateQuantizer->QuantizeVector(newState);
				newIndex = _stateQuantizer->GetIndexFromVector(quantizedNewState);
			}

			_transitions[index] = newIndex;
		}
	}


	// Computes the winning set for which the controller currently is able to adhere to the control specification
	void Verifier::ComputeWinningSet() {
		auto stateSpaceCardinality = _stateQuantizer->GetCardinality();

		// Reset the winning domain
		delete[] _winningSet;
		_winningSet = new bool[stateSpaceCardinality];

		// Define initial winning domain for the fixed point operator
		for (long index = 0; index < stateSpaceCardinality; index++) {
			switch (_specification->GetSpecificationType()) {
			case ControlSpecificationType::Invariance:
				_winningSet[index] = true;
				break;

			case ControlSpecificationType::Reachability:
				auto state = _stateQuantizer->GetVectorFromIndex(index);
				if (_specification->IsInSpecificationSet(state)) _winningSet[index] = true; // 18040 is true
				else _winningSet[index] = false;
				break;
			}
		}

		// Perform fixed algorithm operator on winning set to determine the winning set
		bool iterationConverged = false;
		int iterations = 0;
		while (!iterationConverged) {
			bool setHasChanged = false;
			iterations++;

			for (long index = 0; index < stateSpaceCardinality; index++) {
				auto newState = _transitions[index];
				bool newStateWinningSet = (newState != -1) ? _winningSet[newState] : false;

				if (newState >= 18040) {
					newStateWinningSet = true;
				}

				switch (_specification->GetSpecificationType()) {
				case ControlSpecificationType::Invariance:
					if (!newStateWinningSet) {
						auto changed = SetWinningDomain(index, false);
						if (!setHasChanged) setHasChanged = changed;
					}
					break;

				case ControlSpecificationType::Reachability:
					if (newStateWinningSet) {
						auto changed = SetWinningDomain(index, true);
						if (!setHasChanged) setHasChanged = changed;
					}
					break;
				}
			}

			// DEBUG: Print a map of the set to depict its evolution
			/*for (long index = 0; index < stateSpaceCardinality; index++) {
				auto newState = _transitions[index];
				bool newStateWinningSet = (newState != -1) ? _winningSet[newState] : false;

				if (index % 8 == 0) std::cout << std::endl;

				if (newStateWinningSet) std::cout << "T";
				else std::cout << "F";
			}
			std::cout << std::endl;*/

			if (!setHasChanged) iterationConverged = true;
		}

		std::cout << std::endl << "Iterations: " << iterations << std::endl;
	}


	// Returns the size of the winning set compared to the cardinality of the state space
	long Verifier::GetWinningSetSize() {
		long size = 0;

		for (long index = 0; index < _stateQuantizer->GetCardinality(); index++) {
			if (_winningSet[index]) size++;
		}

		return size;
	}


	// Sets a part of the winning domain, returns whether or not that element has changed
	bool Verifier::SetWinningDomain(long index, bool value) {
		auto currentValue = _winningSet[index];
		if (currentValue != value) {
			_winningSet[index] = value;
			return true;
		}
		else {
			return false;
		}
	}
}