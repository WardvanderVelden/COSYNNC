#include "Verifier.h"

namespace COSYNNC {
	Verifier::Verifier(Plant* plant, Controller* controller, Quantizer* stateQuantizer, Quantizer* inputQuantizer) {
		_plant = plant;
		_controller = controller;
		_stateQuantizer = stateQuantizer;
		_inputQuantizer = inputQuantizer;

		_specification = controller->GetControlSpecification();

		// Initialize soft constants
		_spaceDimension = _stateQuantizer->GetSpaceDimension();
		_spaceCardinality = _stateQuantizer->GetCardinality();
		_spaceEta = _stateQuantizer->GetSpaceEta();
		
		_inputDimension = _inputQuantizer->GetSpaceDimension();
		_inputCardinality = _inputQuantizer->GetCardinality();

		_amountOfVerticesPerCell = pow(2.0, (double)_spaceDimension);
		_amountOfEdgesPerCell = _spaceDimension * pow(2.0, (double)_spaceDimension - 1);

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
		// Free up previous transitions
		delete[] _transitions;
		_transitions = new Transition[_spaceCardinality];

		if (_spaceCardinality > 10000) std::cout << "\t\t";
		for (long index = 0; index < _spaceCardinality; index++) {
			// Print status to monitor progression
			if (_spaceCardinality > 10000 && index % ((long)floor(_spaceCardinality / 20)) == 0) std::cout << (float)((float)index / (float)_spaceCardinality * 100.0) << "% . ";
		
			auto input = _controller->GetInputFromIndex(index);

			ComputeTransitionFunctionForIndex(index, input);
		}
		if (_spaceCardinality > 10000) std::cout << std::endl;
	}

	// Computes the transition function for a single index
	void Verifier::ComputeTransitionFunctionForIndex(long index, Vector input) {
		_transitions[index] = Transition(index);

		auto state = _stateQuantizer->GetVectorFromIndex(index);

		_plant->SetState(state);
		auto newState = _plant->EvaluateOverApproximation(input);

		// Check if newState is in bound, if not we know the transition already
		if (!_stateQuantizer->IsInBounds(newState)) {
			_transitions[index].AddEnd(-1);
			return;
		}

		// Evolve the vertices of the hyper cell to determine the new hyper cell
		auto vertices = OverApproximateEvolution(state, input);

		// TODO: Floodfill through all the dimensions in order to get all the transitions
		// Get the edges between all the vertices
		auto edges = GetEdgesBetweenVertices(vertices);
		for (unsigned int i = 0; i < _amountOfEdgesPerCell; i++) {
			auto edge = edges[_amountOfEdgesPerCell];

			AddEdgeToTransitions(&edge, index);
		}
		
		// TEMPORARY: Over-approximate the evolved hyper cell in order to each transition calculation
		// Find upper and lower bound that over-approximates the over-approximation
		/*Vector lowerBoundVertex = newState;
		Vector upperBoundVertex = newState;

		for (unsigned int i = 0; i < _amountOfVerticesPerCell; i++) {
			auto vertex = vertices[i];
			for (unsigned int j = 0; j < _spaceDimension; j++) {
				lowerBoundVertex[j] = min(lowerBoundVertex[j], vertex[j]);
				upperBoundVertex[j] = max(upperBoundVertex[j], vertex[j]);
			}
		}
		lowerBoundVertex = _stateQuantizer->QuantizeVector(lowerBoundVertex);
		upperBoundVertex = _stateQuantizer->QuantizeVector(upperBoundVertex);

		// Find the amount of cells per dimension axis
		Vector cellsPerDimension(_spaceDimension);
		unsigned long amountOfCells = 1;
		for (unsigned int i = 0; i < _spaceDimension; i++) {
			float delta = upperBoundVertex[i] - lowerBoundVertex[i];
			cellsPerDimension[i] = round(delta / _spaceEta[i]) + 1;
			amountOfCells *= cellsPerDimension[i];
		}

		// Add all the cells in the over-approximation of the new hyper cell as transitions
		Vector currentCell = lowerBoundVertex;
		for (unsigned int i = 0; i < amountOfCells; i++) {
			if (i != 0) {
				for (unsigned int j = 1; j < _spaceDimension; j++) {
					auto modulus = (i % (unsigned int)cellsPerDimension[j - 1]);
					if (modulus == 0) {
						currentCell[j] += _spaceEta[j];
						for (unsigned int k = 0; k < j; k++) {
							currentCell[k] = lowerBoundVertex[k];
						}
						break;
					}
				}
			}

			long end = _stateQuantizer->GetIndexFromVector(_stateQuantizer->QuantizeVector(currentCell));
			_transitions[index].AddEnd(end);

			currentCell[0] += _spaceEta[0];
		}*/

		// Free up memory
		delete[] vertices;
		delete[] edges;
	}


	// Computes the winning set for which the controller currently is able to adhere to the control specification
	void Verifier::ComputeWinningSet() {
		// Free up previous winning set
		delete[] _winningSet;
		_winningSet = new bool[_spaceCardinality] { false };

		// Define initial winning domain for the fixed point operator
		for (long index = 0; index < _spaceCardinality; index++) {
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
		const int indexDivider = round((_stateQuantizer->GetSpaceUpperBound()[0] - _stateQuantizer->GetSpaceLowerBound()[0]) / _spaceEta[0]); // Temporary divider for 2D systems
		if (indexDivider > 250) _verboseMode = false;

		if (_verboseMode) {
			for (long index = 0; index < _spaceCardinality; index++) {
				if (index % indexDivider == 0) std::cout << std::endl << "\t";
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

			if (_spaceCardinality > 10000) {
				std::cout << "\t i: " << iterations << std::endl;
			}

			for (long index = 0; index < _spaceCardinality; index++) {
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
			if (_verboseMode) {
				for (long index = 0; index < _spaceCardinality; index++) {
					if (index % indexDivider == 0) std::cout << std::endl << "\t";
					if (_winningSet[index]) std::cout << "X";
					else std::cout << ".";
				}
				std::cout << std::endl;
			}

			if (!setHasChanged) iterationConverged = true;
		}

		DetermineLosingSet();

		if(_verboseMode) std::cout << std::endl << "\tFixed point iterations: " << iterations << std::endl;

		_winningDomainPercentage = (float)GetWinningSetSize() / (float)_stateQuantizer->GetCardinality() * 100;
	}


	// Determines the losing set and the set of losing cells which are next to the winning domain
	void Verifier::DetermineLosingSet() {
		// Clear losing indices
		_losingIndices.clear();
		_losingWinningNeighborIndices.clear();

		// Determine a list of all the neighboring directions
		auto spaceDimension = _stateQuantizer->GetSpaceDimension();
		auto spaceEta = _stateQuantizer->GetSpaceEta();

		vector<Vector> directions;
		for (unsigned int i = 0; i < spaceDimension; i++) {
			auto dir = Vector(spaceDimension);

			dir[i] = -spaceEta[i];
			directions.push_back(dir);
			directions.push_back(dir * -1);
		}

		// Find losing indices
		for (long index = 0; index < _spaceCardinality; index++) {
			if (!_winningSet[index]) {
				_losingIndices.push_back(index);

				auto state = _stateQuantizer->GetVectorFromIndex(index);
				for (unsigned int i = 0; i < directions.size(); i++) {
					auto neighbor = state + directions[i];
					if (_stateQuantizer->IsInBounds(neighbor)) {
						auto neighborIndex = _stateQuantizer->GetIndexFromVector(neighbor);

						if (IsIndexInWinningSet(neighborIndex)) {
							_losingWinningNeighborIndices.push_back(neighborIndex);
						}
					}
				}
			}
		}
	}


	//  Prints a verbose walk of the current controller using greedy inputs
	void Verifier::PrintVerboseWalk(Vector initialState) {
		_plant->SetState(initialState);

		bool continueWalk = true;
		for (unsigned int i = 0; i < _maxSteps; i++) {
			auto state = _plant->GetState();
			if (!_stateQuantizer->IsInBounds(state)) break;

			auto quantizedState = _stateQuantizer->QuantizeVector(state);

			auto networkOutputDimension = (int)_controller->GetNeuralNetwork()->GetLabelDimension();
			Vector networkOutput(networkOutputDimension);
			auto input = _controller->GetControlAction(quantizedState, &networkOutput);

			_plant->Evolve(input);

			auto newState = _plant->GetState();
			auto satisfied = _specification->IsInSpecificationSet(newState);

			auto displayedOutputs = (networkOutputDimension < 5) ? networkOutputDimension : 5;

			std::cout << "\ti: " << i << "\tx0: " << newState[0] << "\tx1: " << newState[1] << "\tu: " << input[0] << "\ts: " << satisfied;
			for (unsigned int i = 0; i < displayedOutputs; i++) {
				std::cout << "\tn" << i << ": " << networkOutput[i];
			}
			std::cout << std::endl;


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


	// Returns the last calculated percentage of the winning domain compared to the state space
	float Verifier::GetWinningDomainPercentage() {
		return _winningDomainPercentage;
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


	// Sets the verbose mode
	void Verifier::SetVerboseMode(bool verboseMode) {
		_verboseMode = verboseMode;
	}


	// Get a random vector from the space of the losing domain
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


	// Get a random vector from the set of losing states which neighbor winning states
	Vector Verifier::GetVectorFromLosingNeighborDomain() {
		auto losingNeighborIndicesSize = _losingWinningNeighborIndices.size();

		if (losingNeighborIndicesSize > 0) {
			long randomIndex = floor(((float)rand() / RAND_MAX) * (losingNeighborIndicesSize - 1));
			return _stateQuantizer->GetVectorFromIndex(_losingWinningNeighborIndices[randomIndex]);
		}
		else {
			return _stateQuantizer->GetRandomVector();
		}
	}


	// Over approximates all the vertices and returns an array of the new vertices
	Vector* Verifier::OverApproximateEvolution(Vector state, Vector input) {
		auto vertices = _stateQuantizer->GetHyperCellVertices(state);

		Vector* newStateVertices = new Vector[_amountOfVerticesPerCell];

		for (unsigned int i = 0; i < _amountOfVerticesPerCell; i++) {
			auto vertex = vertices[i];

			_plant->SetState(vertex);
			auto newStateVertex = _plant->EvaluateOverApproximation(input);

			newStateVertices[i] = newStateVertex;
		}

		delete[] vertices;

		return newStateVertices;
	}


	// Returns the edges between a set of vertices if the vertices are properly sorted
	Edge* Verifier::GetEdgesBetweenVertices(Vector* vertices) {
		Edge* edges = new Edge[_amountOfEdgesPerCell];

		unsigned int edgeIndex = 0;
		unsigned int edgesAccountedFor = 0;

		for (unsigned int i = 0; i < _spaceDimension; i++) {
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

				auto transposeVector = Vector(_spaceDimension);
				transposeVector[i] += _spaceEta[i];

				transposedEdge.Transpose(transposeVector);

				edges[edgeIndex++] = Edge(transposedEdge);
			}
		}

		return edges;
	}


	// Walks over a single edge and adds all the cells it crosses to the transitions for flood filling
	void Verifier::AddEdgeToTransitions(Edge* edge, unsigned long index) {
		auto direction = edge->GetDirection();

		auto currentPoint = edge->GetStart();
		auto currentCellIndex = _stateQuantizer->GetIndexFromVector(currentPoint);
		auto currentCellCenter = _stateQuantizer->GetVectorFromIndex(currentCellIndex);
		_transitions[index].AddEnd(currentCellIndex);

		auto endCellIndex = _stateQuantizer->GetIndexFromVector(edge->GetEnd());

		while (currentCellIndex != endCellIndex) {
			// Project current point to edge and see if it lays in the edge
			for (unsigned int dim = 0; dim < _spaceDimension; dim++) {
				float edgePosInDim = currentCellCenter[dim] - _spaceEta[dim] * 0.5;
				float distanceToEdge = abs(edgePosInDim - currentPoint[dim]);

				Vector projectedPos = (1 / currentPoint[dim]) * distanceToEdge;
			}

			// Update current cell


			_transitions[index].AddEnd(currentCellIndex);
		}
	}


	// Returns whether or not an index is in the winning domain
	bool Verifier::IsIndexInWinningSet(unsigned long index) {
		if (index < 0 || index > _stateQuantizer->GetCardinality()) return false;
	
		return _winningSet[index];
	}


	// Sets whether or not to use the over approximation for verifier
	void Verifier::SetUseOverApproximation(bool value) {
		_useOverApproximation = value;
	}
}