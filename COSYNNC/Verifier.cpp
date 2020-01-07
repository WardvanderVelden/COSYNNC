#include "Verifier.h"

namespace COSYNNC {
	Verifier::Verifier(Plant* plant, Controller* controller, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		_plant = plant;
		_controller = controller;
		_stateQuantizer = stateQuantizer;
		_inputQuantizer = inputQuantizer;

		_specification = controller->GetControlSpecification();

		// Initialize transitions and winning set
		auto stateSpaceCardinality = _stateQuantizer->GetCardinality();

		_transitions = new long[stateSpaceCardinality] { -1 };
		_winningSet = new bool[stateSpaceCardinality] { false };
	}


	Verifier::~Verifier() {
		delete[] _transitions;
		delete[] _winningSet;

		//for (long index = 0; index < _stateQuantizer->GetCardinality(); index++) delete _transitions[index];
	}


	// Calculates the transition function that transitions any state in the state space to a set of states in the state space based on the control law
	void Verifier::ComputeTransitionFunction() {
		for (long index = 0; index < _stateQuantizer->GetCardinality(); index++) {
			// Get state based on the index
			auto state = _stateQuantizer->GetVectorFromIndex(index);

			// Calculate transition for every vertex in the hyper cell
			auto vertices = _stateQuantizer->GetHyperCellVertices(state);
			Vector newStateVertices[vertices.size()];
			for(unsigned int i = 0; i < _spaceDim * 2; i++) {
				auto vertex = vertices[i];

				_plant->SetState(vertex);

				auto input = _controller->GetControlAction(state);
				auto newStateVertex = _plant->StepOverApproximation(input);

				newStateVertices[i] = newStateVertex;
			}

			// Go through the new state vertices to determine all the new states that are captured by it
			for(unsigned int i = 0; i < _spaceDim; i++) {
				// Vertices are in the same cell
				if(newStateVertices[i * 2] == newStateVertices[i * 2 + 1]) {
					_transitions[index] = _stateQuantizer->QuantizeVector(newStateVertices[i]);
				}
			}

			// Interpolate between the vertices to find all the new states and add these to an array of transitions
			

			// Find the input based on the current controller
			/*auto input = _controller->GetControlAction(state);

			// Calculate the transition
			_plant->Evolve(input);
			auto newState = _plant->GetState();

			long newIndex = -1;
			if (_stateQuantizer->IsInBounds(newState)) {
				auto quantizedNewState = _stateQuantizer->QuantizeVector(newState);
				newIndex = _stateQuantizer->GetIndexFromVector(quantizedNewState);
			}

			_transitions[index] = newIndex;*/
		}
	}


	// Computes the winning set for which the controller currently is able to adhere to the control specification
	void Verifier::ComputeWinningSet() {
		auto stateSpaceCardinality = _stateQuantizer->GetCardinality();

		bool verbose = true;

		// Define initial winning domain for the fixed point operator
		for (long index = 0; index < stateSpaceCardinality; index++) {
			auto state = _stateQuantizer->GetVectorFromIndex(index);

			switch (_specification->GetSpecificationType()) {
			case ControlSpecificationType::Invariance:
				_winningSet[index] = _specification->IsInSpecificationSet(state);

				break;

			case ControlSpecificationType::Reachability:
				if (_specification->IsInSpecificationSet(state)) _winningSet[index] = true;
				else _winningSet[index] = false;
			
				break;
			}
		}

		// DEBUG: Print a map of the set to depict its evolution
		if (verbose) {
			for (long index = 0; index < stateSpaceCardinality; index++) {
				if (index % 32 == 0) std::cout << std::endl;
				if (_winningSet[index]) std::cout << "X";
				else std::cout << ".";
			}
			std::cout << std::endl;
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

				switch (_specification->GetSpecificationType()) {
				case ControlSpecificationType::Invariance:
					if (!_winningSet[index]) break;

					if (!newStateWinningSet) {
						auto changed = SetWinningDomain(index, false);
						if (!setHasChanged) setHasChanged = changed;
					}
					break;

				case ControlSpecificationType::Reachability:
					if (_winningSet[index]) break;

					if (newStateWinningSet) {
						auto changed = SetWinningDomain(index, true);
						if (!setHasChanged) setHasChanged = changed;
					}
					break;
				}
			}

			// DEBUG: Print a map of the set to depict its evolution
			if (verbose) {
				for (long index = 0; index < stateSpaceCardinality; index++) {
					if (index % 32 == 0) std::cout << std::endl;
					if (_winningSet[index]) std::cout << "X";
					else std::cout << ".";
				}
				std::cout << std::endl;
			}

			if (!setHasChanged) iterationConverged = true;
		}

		// Collect losing indices for training purposes
		_losingIndices.clear();
		for (long index = 0; index < stateSpaceCardinality; index++) {
			if (!_winningSet[index]) _losingIndices.push_back(index);
		}

		std::cout << std::endl << "Fixed point iterations: " << iterations << std::endl;
	}


	//  Prints a verbose walk of the current controller using greedy inputs
	void Verifier::PrintVerboseWalk(Vector initialState) {
		_plant->SetState(initialState);

		bool continueWalk = true;
		for (unsigned int i = 0; i < _maxSteps; i++) {
			auto state = _plant->GetState();
			if (!_stateQuantizer->IsInBounds(state)) break;

			auto quantizedState = _stateQuantizer->QuantizeVector(state);

			auto input = _controller->GetControlAction(quantizedState);

			_plant->Evolve(input);

			auto newState = _plant->GetState();
			//auto quantizedNewState = _stateQuantizer->QuantizeVector(newState);
			auto satisfied = _specification->IsInSpecificationSet(newState);

			std::cout << "\ti: " << i << "\tx0: " << newState[0] << "\tx1: " << newState[1] << "\tu: " << input[0] << "\ts: " << satisfied << std::endl;
			
			switch (_specification->GetSpecificationType()) {
			case ControlSpecificationType::Invariance:
				if (!satisfied) continueWalk = false;
				break;
			case ControlSpecificationType::Reachability:
				if (satisfied) continueWalk = false;
				break;
			}

			if (!continueWalk) break;
		}
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


	// Get a random vector from the space of the lossing domain
	Vector Verifier::GetVectorFromLosingDomain() {
		auto losingIndicesSize = _losingIndices.size();

		if (losingIndicesSize > 0) {
			long randomIndex = floor(((float)rand() / RAND_MAX) * (losingIndicesSize - 1));
			return _stateQuantizer->GetVectorFromIndex(_losingIndices[randomIndex]);
		}
		else {
			return _stateQuantizer->GetRandomVector();
		}
	}


	// Get a random vector in a radius to the goal based on training time
	Vector Verifier::GetVectorRadialFromGoal(float progression) {
		auto stateDim = _stateQuantizer->GetSpaceDimension();
		Vector vector(stateDim);
		
		auto goal = _specification->GetCenter();
		auto lowerBound = _stateQuantizer->GetSpaceLowerBound();
		auto upperBound = _stateQuantizer->GetSpaceUpperBound();

		for (int i = 0; i < stateDim; i++) {
			float deltaLower = goal[i] - lowerBound[i];
			float deltaUpper = upperBound[i] - goal[i];

			float spaceSpan = deltaLower + deltaUpper;
			float randomValue = ((float)rand() / RAND_MAX);

			float lowerSpaceProbability = (deltaLower / spaceSpan);

			if (randomValue < lowerSpaceProbability) {
				randomValue = randomValue /  lowerSpaceProbability;
				vector[i] = goal[i] - deltaLower * randomValue * progression;
			}
			else {
				randomValue = (randomValue - lowerSpaceProbability) / (1.0 - lowerSpaceProbability);
				vector[i] = goal[i] + deltaUpper * randomValue * progression;
			}
		}

		return vector;
	}
}