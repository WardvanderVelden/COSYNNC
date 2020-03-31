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

		CalculateVerticesOnHyperplaneDistribution();

		// Initialize transitions and winning set
		auto stateSpaceCardinality = _stateQuantizer->GetCardinality();

		_transitions = new Transition[stateSpaceCardinality];
		for (unsigned long index = 0; index < stateSpaceCardinality; index++) _transitions[index] = Transition(index);

		_winningSet = new bool[stateSpaceCardinality] { false };
	}


	Verifier::~Verifier() {
		delete[] _transitions;
		delete[] _winningSet;
	}


	// Calculates the transition function that transitions any state in the state space to a set of states in the state space based on the control law
	void Verifier::ComputeTransitionFunction() {
		std::cout << "\t\t";
		for (long index = 0; index < _spaceCardinality; index++) {
			// Print status to monitor progression
			if (index % ((long)floor(_spaceCardinality / 20)) == 0) std::cout << (float)((float)index / (float)_spaceCardinality * 100.0) << "% . ";
		
			auto input = _controller->GetInputFromIndex(index);

			ComputeTransitionFunctionForIndex(index, input);
		}
		std::cout << std::endl;
	}


	// Computes the transition function for a single index
	void Verifier::ComputeTransitionFunctionForIndex(long index, Vector input) {
		if (_transitions[index].HasInputChanged(input)) {
			auto state = _stateQuantizer->GetVectorFromIndex(index);
			_plant->SetState(state);

			auto newState = _plant->EvaluateOverApproximation(input);

			// Check if newState is in bounds of the state space, if not then we know the transition already
			if (!_stateQuantizer->IsInBounds(newState)) {
				_transitions[index].AddEnd(-1);
				return;
			}

			// Evolve the vertices of the cell and find the resulting hyperplanes
			vector<Hyperplane> hyperplanes;
			auto vertices = OverApproximateEvolution(state, input, hyperplanes);

			// Get the point in space that represents the center of the vertices
			auto center = Vector(_spaceDimension);
			for (unsigned int i = 0; i < _amountOfVerticesPerCell; i++) {
				center = center + vertices[i];
			}
			center = center * (1.0 / (float)_amountOfVerticesPerCell);

			// Flood fill as long as a vertex in a cell is in the internal area
			FloodfillBetweenHyperplanes(index, center, hyperplanes);

			// Free up memory
			delete[] vertices;
		}
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
			iterations++;
			std::cout << "\t i: " << iterations;

			auto setHasChanged = PerformFixedPointIteration();

			// DEBUG: Print a map of the set to depict its evolution
			if (_verboseMode) {
				for (long index = 0; index < _spaceCardinality; index++) {
					if (index % indexDivider == 0) std::cout << std::endl << "\t";
					if (_winningSet[index]) std::cout << "X";
					else std::cout << ".";
				}
				std::cout << std::endl;
			}

			if (!setHasChanged) {
				if (ValidateDomain()) {
					iterationConverged = true;
				}
				else {
					std::cout << " discrepancy detected";
					//iterationConverged = true;
				}
			}

			std::cout << std::endl;
		}

		// DEBUG: Find winning set and look at transitions there
		for (unsigned long index = 0; index < _spaceCardinality; index++) {
			if (IsIndexInWinningSet(index)) {
				std::cout << std::endl << "\t" << index << " is in winning set, transitions are: " << std::endl;

				auto ends = _transitions[index].GetEnds();
				for (unsigned int i = 0; i < ends.size(); i++) {
					std::cout << "\t" << ends[i] << " winning: " << IsIndexInWinningSet(ends[i]) << std::endl;
				}

				break;
			}
		}

		DetermineLosingSet();

		if(_verboseMode) std::cout << std::endl << "\tFixed point iterations: " << iterations << std::endl;

		_winningDomainPercentage = (float)GetWinningSetSize() / (float)_stateQuantizer->GetCardinality() * 100;
	}


	// Performs a single fixed point iteration
	bool Verifier::PerformFixedPointIteration() {
		bool setHasChanged = false;

		for (long index = 0; index < _spaceCardinality; index++) {
			// Determine if the transition always ends in the winning set
			auto ends = _transitions[index].GetEnds();

			bool alwaysEndsInWinningSet = true;
			if (ends.size() == 0) alwaysEndsInWinningSet = false;

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

		return setHasChanged;
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
	Vector* Verifier::OverApproximateEvolution(Vector state, Vector input, vector<Hyperplane>& hyperplanes) {
		// Get the vertices that make up the hypercell
		auto vertices = _stateQuantizer->GetCellVertices(state);

		// Get the hyperplanes that are naturally arise between the vertices and set the normal
		hyperplanes = GetHyperplanesBetweenVertices(vertices, state);

		// Over approximate the dynamics of the plant and update vertices
		for (unsigned int i = 0; i < _amountOfVerticesPerCell; i++) {
			auto vertex = vertices[i];

			_plant->SetState(vertex);
			vertices[i] = _plant->EvaluateOverApproximation(input);
		}

		// Over approximate the dynamics of the plant to update the normal vectors
		for (unsigned int i = 0; i < hyperplanes.size(); i++) {
			hyperplanes[i].OverApproximateNormal(_plant, input);
		}
				
		return vertices;
	}


	// Returns the hyperplanes that naturally arise between the vertices
	vector<Hyperplane> Verifier::GetHyperplanesBetweenVertices(Vector* vertices, Vector cellCenter) {
		vector<Hyperplane> hyperplanes;

		for (unsigned int i = 0; i < (_spaceDimension * 2); i++) {
			Hyperplane hyperplane(_spaceDimension);

			// Assign appropriate vertices to the appropriate hyperplanes based on the precalculation of the vertex distribution
			auto verticesOnHyperplane = _verticesOnHyperplaneDistribution[i];

			for (unsigned int j = 0; j < verticesOnHyperplane.size(); j++) {
				hyperplane.AddPointToHyperplane(&vertices[verticesOnHyperplane[j]]);
			}

			// Define the normal based on the precalculation thereof
			auto normal = _normalsOfHyperplane[i];
			hyperplane.SetNormal(normal, cellCenter);

			// Add hyperplane
			hyperplanes.push_back(hyperplane);
		}

		return hyperplanes;
	}


	// Flood fills between planes, adding the indices of the cells to the transitions of the origin cell
	void Verifier::FloodfillBetweenHyperplanes(unsigned long index, Vector center, vector<Hyperplane>& planes) {
		unsigned long centerIndex = _stateQuantizer->GetIndexFromVector(center);

		// Generate initial floodfill order
		vector<unsigned long> indices;
		vector<unsigned long> processedIndices;
		indices.push_back(centerIndex);

		// Process all the floodfill orders
		while (indices.size() != 0) {
			unsigned long currentIndex = *indices.begin();

			// Test if their is a vertex of the current cell that is within the planes
			auto vertices = _stateQuantizer->GetCellVertices(currentIndex);

			bool isBetweenPlanes = false;
			for (unsigned int i = 0; i < _amountOfVerticesPerCell; i++) {
				if (IsPointBetweenHyperplanes(vertices[i], planes)) {
					isBetweenPlanes = true;
					break;
				}
			}

			// TEMPORARY: Always add the center index to the transition
			if (isBetweenPlanes || currentIndex == centerIndex) {
				// Add order to transitions
				_transitions[index].AddEnd(currentIndex);

				// Generate new orders that branch from current order
				auto cellCenter = _stateQuantizer->GetVectorFromIndex(currentIndex);
				for (unsigned int i = 0; i < _spaceDimension; i++) {
					Vector direction(_spaceDimension);

					// Lower direction
					direction[i] = -1.0;
					AddFloodfillOrder(cellCenter, direction, indices, processedIndices);
									
					// Upper direction
					direction[i] = 1.0;
					AddFloodfillOrder(cellCenter, direction, indices, processedIndices);
				}
			}

			// Track which indices have already been processed
			processedIndices.push_back(currentIndex);
			indices.erase(indices.begin(), indices.begin() + 1);

			// Clear memory
			delete[] vertices;
		}

		// Clear to prevent leaking memory
		indices.clear();
		processedIndices.clear();
	}


	// Generates the appropriate floodfill indices based on the current inex and the processed indices
	void Verifier::AddFloodfillOrder(Vector center, Vector direction, vector<unsigned long>& indices, vector<unsigned long>& processedIndices) {
		// Get index of neighbor cell that is in that direction
		auto neighborCell = center + (direction * _spaceEta);
		auto neighborCellIndex = _stateQuantizer->GetIndexFromVector(neighborCell);

		// If the neighbor cell index is -1 it is out of bounds and we do not need to add it
		if (neighborCellIndex == -1) return;

		// Check if cell is not already processed
		for (auto processedIndex : processedIndices) {
			if (processedIndex == neighborCellIndex) return;
		}

		indices.push_back(neighborCellIndex);
	}


	// Checks if a point is contained between planes
	bool Verifier::IsPointBetweenHyperplanes(Vector point, vector<Hyperplane>& planes) {
		for (unsigned int i = 0; i < planes.size(); i++) {
			if (!planes[i].IsPointOnInternalSide(point)) return false;
		}

		return true;
	}


	// Calculates the vertices to hyperplane distribution
	void Verifier::CalculateVerticesOnHyperplaneDistribution() {
		auto cellCenter = _stateQuantizer->GetVectorFromIndex(0);
		auto vertices = _stateQuantizer->GetCellVertices(0);

		for (unsigned int dim = 0; dim < _spaceDimension; dim++) {
			Vector normal(_spaceDimension);

			// Process both sides
			for (float side = -1.0; side <= 1.0; side += 1.0) {
				if (side == 0.0) continue;

				// Find vertices that are on the hyperplane
				vector<unsigned short> verticesOnHyperplane;
				for (unsigned int i = 0; i < _amountOfVerticesPerCell; i++) {
					auto vertex = vertices[i];

					if ((vertex[dim] * side) > (cellCenter[dim] * side)) {
						verticesOnHyperplane.push_back(i);
					}
				}

				// Define normal
				normal[dim] = side * -1.0;

				// Add normal to distribution
				_verticesOnHyperplaneDistribution.push_back(verticesOnHyperplane);
				_normalsOfHyperplane.push_back(normal);
			}
		}

		delete[] vertices;
	}


	// Returns whether or not an index is in the winning domain
	bool Verifier::IsIndexInWinningSet(unsigned long index) {
		if (index < 0 || index > _stateQuantizer->GetCardinality()) return false;
	
		return _winningSet[index];
	}



	// TEMPORARY: Validation method in order to verify and bugfix the behaviour of the verifier
	bool Verifier::ValidateDomain() {
		bool hasDiscrepancy = false;

		auto specificationType = _specification->GetSpecificationType();
		
		for (long index = 0; index < _spaceCardinality; index++) {
			bool isLosingHole = true; // This will remain true if there is a losing state surrounded by winning states
			bool isWinningIsland = true; // This will remain true if the is a winning state surrounded by losing states

			for (unsigned int dim = 0; dim < _spaceDimension; dim++) {
				auto cell = _stateQuantizer->GetVectorFromIndex(index);

				for (int i = -1; i <= 1; i++) {
					if (i == 0) continue;

					Vector neighbor = cell;
					neighbor[dim] = cell[dim] + (_spaceEta[dim] * i);

					auto neighborIndex = _stateQuantizer->GetIndexFromVector(neighbor);
					auto neighborWinning = IsIndexInWinningSet(neighborIndex);

					if (!neighborWinning) isLosingHole = false;
					if (neighborWinning) isWinningIsland = false;
				}
			}

			// Handle winning islands and losing holes
			if (isLosingHole && specificationType == ControlSpecificationType::Reachability) {
				if (_plant->GetIsLinear()) _winningSet[index] = true;
				hasDiscrepancy = true;
			}

			if (isWinningIsland && specificationType == ControlSpecificationType::Invariance) {
				if (_plant->GetIsLinear()) _winningSet[index] = false;
				hasDiscrepancy = true;
			}
		}

		if (hasDiscrepancy) return false;
		return true;
	}
}