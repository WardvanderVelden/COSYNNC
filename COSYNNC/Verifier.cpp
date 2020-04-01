#include "Verifier.h"

namespace COSYNNC {
	// Constructor that setup up the verifier for use
	Verifier::Verifier(Abstraction* abstraction) {
		_abstraction = abstraction;

		_winningSet = new bool[_abstraction->GetStateQuantizer()->GetCardinality()]{ false };
	}


	Verifier::~Verifier() {
		delete[] _winningSet;
	}


	// Calculates the transition function that transitions any state in the state space to a set of states in the state space based on the control law
	void Verifier::ComputeTransitions() {
		const auto spaceCardinality = _abstraction->GetStateQuantizer()->GetCardinality();

		std::cout << "\t\t";
		for (long index = 0; index < spaceCardinality; index++) {
			// Print status to monitor progression
			if (index % ((long)floor(spaceCardinality / 20)) == 0) std::cout << (float)((float)index / (float)spaceCardinality * 100.0) << "% . ";
		
			auto input = _abstraction->GetController()->GetInputFromIndex(index);

			_abstraction->ComputeTransitionFunctionForIndex(index, input);
		}
		std::cout << std::endl;
	}


	// Computes the winning set for which the controller currently is able to adhere to the control specification
	void Verifier::ComputeWinningSet() {
		// Free up previous winning set
		delete[] _winningSet;
		_winningSet = new bool[_abstraction->GetStateQuantizer()->GetCardinality()] { false };

		// Define initial winning domain for the fixed point operator
		auto controlSpecification = _abstraction->GetControlSpecification();

		for (long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
			auto state = _abstraction->GetStateQuantizer()->GetVectorFromIndex(index);

			switch (_abstraction->GetControlSpecification()->GetSpecificationType()) {
			case ControlSpecificationType::Invariance:
				_winningSet[index] = controlSpecification->IsInSpecificationSet(state);

				break;

			case ControlSpecificationType::Reachability:
				if (controlSpecification->IsInSpecificationSet(state)) _winningSet[index] = true;
				else _winningSet[index] = false;
			
				break;
			}
		}

		// DEBUG: Print a map of the set to depict its evolution
		const int indexDivider = round((_abstraction->GetStateQuantizer()->GetUpperBound()[0] - _abstraction->GetStateQuantizer()->GetLowerBound()[0]) / _abstraction->GetStateQuantizer()->GetEta()[0]); // Temporary divider for 2D systems
		if (indexDivider > 250) _verboseMode = false;

		if (_verboseMode) {
			for (long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
				if (index % indexDivider == 0) std::cout << std::endl << "\t";
				if (_winningSet[index]) std::cout << "X";
				else std::cout << ".";
			}
			std::cout << std::endl;
		}
		
		// Perform fixed algorithm operator on winning set to determine the winning set
		bool hasIterationConverged = false;
		unsigned int iterations = 0;
		unsigned int discrepancies = 0;
		while (!hasIterationConverged) {
			iterations++;
			std::cout << "\t i: " << iterations;

			auto hasSetChanged = PerformSingleFixedPointOperation();

			// DEBUG: Print a map of the set to depict its evolution
			if (_verboseMode) {
				for (long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
					if (index % indexDivider == 0) std::cout << std::endl << "\t";
					if (_winningSet[index]) std::cout << "X";
					else std::cout << ".";
				}
				std::cout << std::endl;
			}

			if (!hasSetChanged) {
				if (ValidateDomain()) {
					hasIterationConverged = true;
				}
				else {
					std::cout << " discrepancy detected";
					discrepancies++;
					if (discrepancies >= 15) hasIterationConverged = true;
				}
			}

			std::cout << std::endl;
		}

		// DEBUG: Find winning set and look at transitions there
		for (unsigned long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
			if (IsIndexInWinningSet(index)) {
				std::cout << std::endl << "\t" << index << " is in winning set, transitions are: " << std::endl;

				auto ends = _abstraction->GetTransitionOfIndex(index)->GetEnds();
				for (unsigned int i = 0; i < ends.size(); i++) {
					std::cout << "\t" << ends[i] << " winning: " << IsIndexInWinningSet(ends[i]) << std::endl;
				}

				break;
			}
		}

		DetermineLosingSet();

		if(_verboseMode) std::cout << std::endl << "\tFixed point iterations: " << iterations << std::endl;

		_winningDomainPercentage = (float)GetWinningSetSize() / (float)_abstraction->GetStateQuantizer()->GetCardinality() * 100;
	}


	// Performs a single fixed point iteration
	bool Verifier::PerformSingleFixedPointOperation() {
		bool hasSetChanged = false;

		for (long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
			// Determine if the transition always ends in the winning set
			auto ends = _abstraction->GetTransitionOfIndex(index)->GetEnds();

			bool allEndsInWinningSet = true;
			if (ends.size() == 0) allEndsInWinningSet = false;

			for (auto end : ends) {
				if (end == -1 || !_winningSet[end]) allEndsInWinningSet = false;
			}

			// Handle transitions based on specification
			switch (_abstraction->GetControlSpecification()->GetSpecificationType()) {
			case ControlSpecificationType::Invariance:
				if (!_winningSet[index]) break;

				if (!allEndsInWinningSet) {
					auto changed = SetWinningDomain(index, false);
					if (!hasSetChanged) hasSetChanged = changed;
				}
				break;

			case ControlSpecificationType::Reachability:
				if (_winningSet[index]) break;

				if (allEndsInWinningSet) {
					auto changed = SetWinningDomain(index, true);
					if (!hasSetChanged) hasSetChanged = changed;
				}
				break;
			}
		}

		return hasSetChanged;
	}


	// Determines the losing set and the set of losing cells which are next to the winning domain
	void Verifier::DetermineLosingSet() {
		// Clear losing indices
		_losingIndices.clear();
		_losingWinningNeighborIndices.clear();

		// Determine a list of all the neighboring directions
		auto spaceDimension = _abstraction->GetStateQuantizer()->GetDimension();
		auto spaceEta = _abstraction->GetStateQuantizer()->GetEta();

		vector<Vector> directions;
		for (unsigned int i = 0; i < spaceDimension; i++) {
			auto dir = Vector(spaceDimension);

			dir[i] = -spaceEta[i];
			directions.push_back(dir);
			directions.push_back(dir * -1);
		}

		// Find losing indices
		for (long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
			if (!_winningSet[index]) {
				_losingIndices.push_back(index);

				auto state = _abstraction->GetStateQuantizer()->GetVectorFromIndex(index);
				for (unsigned int i = 0; i < directions.size(); i++) {
					auto neighbor = state + directions[i];
					if (_abstraction->GetStateQuantizer()->IsInBounds(neighbor)) {
						auto neighborIndex = _abstraction->GetStateQuantizer()->GetIndexFromVector(neighbor);

						if (IsIndexInWinningSet(neighborIndex)) {
							_losingWinningNeighborIndices.push_back(neighborIndex);
						}
					}
				}
			}
		}
	}


	// Returns whether or not an index is in the winning domain
	bool Verifier::IsIndexInWinningSet(unsigned long index) {
		if (index < 0 || index > _abstraction->GetStateQuantizer()->GetCardinality()) return false;
	
		return _winningSet[index];
	}


	// TEMPORARY: Validation method in order to verify and bugfix the behaviour of the verifier
	bool Verifier::ValidateDomain() {
		bool hasDiscrepancy = false;

		auto spaceEta = _abstraction->GetStateQuantizer()->GetEta();
		auto specificationType = _abstraction->GetControlSpecification()->GetSpecificationType();
		
		for (long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
			bool isLosingHole = true; // This will remain true if there is a losing state surrounded by winning states
			bool isWinningIsland = true; // This will remain true if the is a winning state surrounded by losing states

			for (unsigned int dim = 0; dim < _abstraction->GetStateQuantizer()->GetDimension(); dim++) {
				auto cell = _abstraction->GetStateQuantizer()->GetVectorFromIndex(index);

				for (int i = -1; i <= 1; i++) {
					if (i == 0) continue;

					Vector neighbor = cell;
					neighbor[dim] = cell[dim] + (spaceEta[dim] * i);

					auto neighborIndex = _abstraction->GetStateQuantizer()->GetIndexFromVector(neighbor);
					auto neighborWinning = IsIndexInWinningSet(neighborIndex);

					if (!neighborWinning) isLosingHole = false;
					if (neighborWinning) isWinningIsland = false;
				}
			}

			// Handle winning islands and losing holes
			if (isLosingHole && specificationType == ControlSpecificationType::Reachability) {
				if (_abstraction->GetPlant()->GetIsLinear()) _winningSet[index] = true;
				hasDiscrepancy = true;
			}

			if (isWinningIsland && specificationType == ControlSpecificationType::Invariance) {
				if (_abstraction->GetPlant()->GetIsLinear()) _winningSet[index] = false;
				hasDiscrepancy = true;
			}
		}

		if (hasDiscrepancy) return false;
		return true;
	}


	#pragma region Getters and Setters

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
			return _abstraction->GetStateQuantizer()->GetVectorFromIndex(_losingIndices[randomIndex]);
		}
		else {
			return _abstraction->GetStateQuantizer()->GetRandomVector();
		}
	}


	// Get a random vector from the set of losing states which neighbor winning states
	Vector Verifier::GetVectorFromLosingNeighborDomain() {
		auto losingNeighborIndicesSize = _losingWinningNeighborIndices.size();

		if (losingNeighborIndicesSize > 0) {
			long randomIndex = floor(((float)rand() / RAND_MAX) * (losingNeighborIndicesSize - 1));
			return _abstraction->GetStateQuantizer()->GetVectorFromIndex(_losingWinningNeighborIndices[randomIndex]);
		}
		else {
			return _abstraction->GetStateQuantizer()->GetRandomVector();
		}
	}


	// Returns the size of the winning set compared to the cardinality of the state space
	long Verifier::GetWinningSetSize() {
		long size = 0;

		for (long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
			if (_winningSet[index]) size++;
		}

		return size;
	}


	// Returns the last calculated percentage of the winning domain compared to the state space
	float Verifier::GetWinningDomainPercentage() {
		return _winningDomainPercentage;
	}

	#pragma endregion Getters and Setters
}