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

		_transitions = new Transition[stateSpaceCardinality];
		_winningSet = new bool[stateSpaceCardinality] { false };
	}


	Verifier::~Verifier() {
		delete[] _transitions;
		delete[] _winningSet;
	}


	// Calculates the transition function that transitions any state in the state space to a set of states in the state space based on the control law
	void Verifier::ComputeTransitionFunction() {
		const auto spaceDim = _stateQuantizer->GetSpaceDimension();

		const unsigned int amountOfVertices = pow(2.0, (double)spaceDim);
		const unsigned int amountOfEdges = spaceDim * pow(2.0, (double)spaceDim - 1);

		for (long index = 0; index < _stateQuantizer->GetCardinality(); index++) {
			_transitions[index] = Transition(index);

			// Get the state that corresponds to the index
			auto state = _stateQuantizer->GetVectorFromIndex(index);

			// Evolve the vertices of the hyper cell to determine the new hyper cell
			auto vertices = OverApproximateEvolution(state);

			// Determine edges of the new vertices
			auto edges = GetEdgesBetweenVertices(vertices);

			// Determine border cells
			

			// Floodfill between the vertices in order to find all transitions


			// TEMPORARY: Add all the vertices to the transitions if they are not yet in the transition
			for (unsigned int i = 0; i < amountOfVertices; i++) {
				auto vertex = vertices[i];

				long end = (_stateQuantizer->IsInBounds(vertex)) ? _stateQuantizer->GetIndexFromVector(_stateQuantizer->QuantizeVector(vertex)) : -1;
				_transitions[index].AddEnd(end);
			}

			// Free up memory
			delete[] vertices;
			delete[] edges;
		}
	}


	// Computes the winning set for which the controller currently is able to adhere to the control specification
	void Verifier::ComputeWinningSet() {
		auto stateSpaceCardinality = _stateQuantizer->GetCardinality();

		bool verboseMode = true;

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
		if (verboseMode) {
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
				// Determine if the transition always ends in the winning set
				auto ends = _transitions[index].GetEnds();

				bool alwaysEndsInWinningSet = true;
				for (auto end : ends) {
					if (end == -1 || !_winningSet[end]) alwaysEndsInWinningSet = false;
				}

				// Handle transitions based on specification
				switch (_specification->GetSpecificationType()) {
				case ControlSpecificationType::Invariance:
					if (!_winningSet[index]) break;

					if (!alwaysEndsInWinningSet) {
						auto changed = SetWinningDomain(index, false);
						if (!setHasChanged) setHasChanged = changed;
					}
					break;

				case ControlSpecificationType::Reachability:
					if (_winningSet[index]) break;

					if (alwaysEndsInWinningSet) {
						auto changed = SetWinningDomain(index, true);
						if (!setHasChanged) setHasChanged = changed;
					}
					break;
				}
			}

			// DEBUG: Print a map of the set to depict its evolution
			if (verboseMode) {
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

		if(verboseMode)	std::cout << std::endl << "Fixed point iterations: " << iterations << std::endl;
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


	// Over approximates all the vertices and returns an array of the new vertices
	Vector* Verifier::OverApproximateEvolution(Vector state) {
		const auto spaceDim = _stateQuantizer->GetSpaceDimension();
		const unsigned int amountOfVertices = pow(2.0, (double)spaceDim);

		auto vertices = _stateQuantizer->GetHyperCellVertices(state);

		Vector* newStateVertices = new Vector[amountOfVertices];

		for (unsigned int i = 0; i < amountOfVertices; i++) {
			auto vertex = vertices[i];

			_plant->SetState(vertex);

			auto input = _controller->GetControlAction(state);
			//auto newStateVertex = _plant->StepOverApproximation(input);
			auto newStateVertex = _plant->StepDynamics(input); // TODO: Switch this for the over approximation dynamics

			newStateVertices[i] = newStateVertex;
		}

		delete[] vertices;

		return newStateVertices;
	}


	// Returns the edges between a set of vertices if the vertices are properly sorted
	Edge* Verifier::GetEdgesBetweenVertices(Vector* vertices) {
		const auto spaceDim = _stateQuantizer->GetSpaceDimension();
		auto spaceEta = _stateQuantizer->GetSpaceEta();

		const unsigned int amountOfEdges = spaceDim * pow(2.0, (double)spaceDim - 1);

		Edge* edges = new Edge[amountOfEdges];

		unsigned int edgeIndex = 0;
		unsigned int edgesAccountedFor = 0;

		for (unsigned int i = 0; i < spaceDim; i++) {
			const unsigned int verticesForDimension = pow(2.0, (double)i + 1);

			edgesAccountedFor = edgeIndex;

			for (unsigned int j = 0; j < verticesForDimension / 2; j++) {
				auto currentVertex = vertices[j];
				auto facingVertex = vertices[j + verticesForDimension / 2];

				edges[edgeIndex++] = Edge(currentVertex, facingVertex);
			}

			// Add edges of lower dimension
			for (unsigned int j = 0; j < edgesAccountedFor; j++) {
				auto transposedEdge = edges[j];

				auto transposeVector = Vector(spaceDim);
				transposeVector[i] += spaceEta[i];

				transposedEdge.Transpose(transposeVector);

				edges[edgeIndex++] = Edge(transposedEdge);
			}
		}

		return edges;
	}
}