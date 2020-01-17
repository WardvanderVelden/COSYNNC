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
		const auto batchSize = _controller->GetNeuralNetwork()->GetBatchSize();
		const long amountOfBatches = ceil(_spaceCardinality / batchSize);

		for (long batch = 0; batch <= amountOfBatches; batch++) {	
			unsigned long indexOffset = batch * batchSize;
			unsigned int currentBatchSize = batchSize;
			if (batch == amountOfBatches) {
				currentBatchSize = _spaceCardinality - indexOffset;
			}

			// Collect all the states in the current batch
			Vector* states = new Vector[currentBatchSize];

			for (unsigned int i = 0; i < currentBatchSize; i++) {
				long index = indexOffset + i;
				states[i] = _stateQuantizer->GetVectorFromIndex(index);
			}

			// Get the corresponding inputs through batch network evaluation
			Vector* inputs = new Vector[currentBatchSize];
			inputs = _controller->GetControlActionInBatch(states, currentBatchSize);

			// DEBUG: Testing if individual input retrieval does work
			/*for (unsigned int i = 0; i < currentBatchSize; i++) {
				inputs[i] = _controller->GetControlAction(states[i]);
			}*/

			// Compute the transition function for all the states in the batch
			for (unsigned int i = 0; i < currentBatchSize; i++) {
				long index = indexOffset + i;
				
				auto state = states[i];
				auto input = inputs[i];
				ComputeTransitionFunctionForIndex(index, states[i], inputs[i]);
			}

			delete[] inputs;
			delete[] states;
		}

		// DEBUG: Go through all the indices 
		/*for (long index = 0; index < _spaceCardinality; index++) {
			auto state = _stateQuantizer->GetVectorFromIndex(index);
			auto input = _controller->GetControlAction(state);

			ComputeTransitionFunctionForIndex(index, state, input);
		}*/
	}

	// Computes the transition function for a single index
	void Verifier::ComputeTransitionFunctionForIndex(long index, Vector state, Vector input) {
		_transitions[index] = Transition(index);

		_plant->SetState(state);
		auto newState = _plant->StepDynamics(input); // TODO: Switch for over approximation dynamics

		// Evolve the vertices of the hyper cell to determine the new hyper cell
		auto vertices = OverApproximateEvolution(state, input);

		// Determine edges of the new vertices
		//auto edges = GetEdgesBetweenVertices(vertices);

		// TODO: Floodfill through all the dimensions in order to get all the transitions
		// TEMPORARY: Over-approximate the evolved hyper cell in order to each transition calculation

		// Find upper and lower bound that over-approximates the over-approximation
		Vector lowerBoundVertex = newState;
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

			long end = (_stateQuantizer->IsInBounds(currentCell)) ? _stateQuantizer->GetIndexFromVector(_stateQuantizer->QuantizeVector(currentCell)) : -1;
			_transitions[index].AddEnd(end);

			currentCell[0] += _spaceEta[0];
		}

		// Temporay: Singular end for new state to bugfix invariance verification
		long end = (_stateQuantizer->IsInBounds(newState)) ? _stateQuantizer->GetIndexFromVector(_stateQuantizer->QuantizeVector(newState)) : -1;
		//_transitions[index].AddEnd(end);

		// Free up memory
		delete[] vertices;
		//delete[] edges;
	}


	// Computes the winning set for which the controller currently is able to adhere to the control specification
	void Verifier::ComputeWinningSet() {
		bool verboseMode = true;

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
		if (indexDivider > 250) verboseMode = false;

		if (verboseMode) {
			for (long index = 0; index < _spaceCardinality; index++) {
				if (index % indexDivider == 0) std::cout << std::endl;
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
			if (verboseMode) {
				for (long index = 0; index < _spaceCardinality; index++) {
					if (index % indexDivider == 0) std::cout << std::endl;
					if (_winningSet[index]) std::cout << "X";
					else std::cout << ".";
				}
				std::cout << std::endl;
			}

			if (!setHasChanged) iterationConverged = true;
		}

		// Collect losing indices for training purposes
		_losingIndices.clear();
		for (long index = 0; index < _spaceCardinality; index++) {
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

			Vector networkOutput((int)_inputCardinality);
			auto input = _controller->GetControlAction(quantizedState, &networkOutput);

			_plant->Evolve(input);

			auto newState = _plant->GetState();
			//auto quantizedNewState = _stateQuantizer->QuantizeVector(newState);
			auto satisfied = _specification->IsInSpecificationSet(newState);

			std::cout << "\ti: " << i << "\tx0: " << newState[0] << "\tx1: " << newState[1] << "\tu: " << input[0] << "\ts: " << satisfied;
			for (unsigned int i = 0; i < (int)_inputCardinality; i++) {
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
		Vector vector(_spaceDimension);
		
		auto goal = _specification->GetCenter();
		auto lowerBound = _stateQuantizer->GetSpaceLowerBound();
		auto upperBound = _stateQuantizer->GetSpaceUpperBound();

		for (int i = 0; i < _spaceDimension; i++) {
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
	Vector* Verifier::OverApproximateEvolution(Vector state, Vector input) {
		auto vertices = _stateQuantizer->GetHyperCellVertices(state);

		Vector* newStateVertices = new Vector[_amountOfVerticesPerCell];

		for (unsigned int i = 0; i < _amountOfVerticesPerCell; i++) {
			auto vertex = vertices[i];

			_plant->SetState(vertex);
			//auto newStateVertex = _plant->StepOverApproximation(input);
			auto newStateVertex = _plant->StepDynamics(input); // TODO: Switch this for the over approximation dynamics

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
}