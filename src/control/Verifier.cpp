#include "Verifier.h"

namespace COSYNNC {
	// Constructor that setup up the verifier for use
	Verifier::Verifier(Abstraction* abstraction) {
		_abstraction = abstraction;
		
		_transitionsInFullAbstraction = _abstraction->GetStateQuantizer()->GetCardinality() * _abstraction->GetInputQuantizer()->GetCardinality();
		_transitionsInAbstraction = 0;

		_winningSet = new bool[_abstraction->GetStateQuantizer()->GetCardinality()]{ false };
	}


	Verifier::~Verifier() {
		delete[] _winningSet;
	}


	// Calculates the transition function that transitions any state in the state space to a set of states in the state space based on the control law
	void Verifier::ComputeTransitions() {
		const auto spaceCardinality = _abstraction->GetStateQuantizer()->GetCardinality();

		// Check if we need to reset transitions because we are not retaining them
		if (!_abstraction->IsSavingTransitions()) _abstraction->EmptyTransitions();

		std::cout << "\t\t";

		// Get the amount of cores that the hardware contains
		auto concurrentThreadsSupported = 1; // thread::hardware_concurrency();
		unsigned long partitionSize = floor((float)spaceCardinality / (float)concurrentThreadsSupported);

		// Setup the threads and start computing transitions
		vector<thread> threadGroup;
		for (unsigned int i = 0; i < concurrentThreadsSupported; i++) {
			unsigned long start = i * partitionSize;
			unsigned long end = (i != (i-1)) ? (i + 1) * partitionSize : spaceCardinality;

			threadGroup.push_back(thread([this, start, end, i] { this->ComputeSubsetOfTransitions(i, start, end); }));
		}

		// Join threads (waiting for threads to complete)
		for (unsigned int i = 0; i < concurrentThreadsSupported; i++) {
			threadGroup.at(i).join();
		}

		//threadGroup[1].join();

		std::cout << std::endl;
	}


	// Computes a subset of the total amount of required transitions, is used for multithreading
	void Verifier::ComputeSubsetOfTransitions(unsigned int thread, unsigned long start, unsigned long end) {
		const auto spaceCardinality = _abstraction->GetStateQuantizer()->GetCardinality();

		for (unsigned long index = start; index < end; index++) {
			if (index % ((long)floor(spaceCardinality / 100)) == 0) {
				std::cout << "." << std::flush; // Print dot monitor progress
			}

			auto input = _abstraction->GetController()->GetControlActionFromIndex(index);
			if (_abstraction->ComputeTransitionFunctionForIndex(index, input)) {
				_transitionsInAbstraction++;
			}
		}
	}


	// Initializes the winning set for verification
	void Verifier::InitializeWinningSet() {
		// Free up previous winning set
		delete[] _winningSet;
		_winningSet = new bool[_abstraction->GetStateQuantizer()->GetCardinality()]{ false };

		// Define initial winning domain for the fixed point operator
		auto controlSpecification = _abstraction->GetControlSpecification();

		for (long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
			auto state = _abstraction->GetStateQuantizer()->GetVectorFromIndex(index);

			_winningSet[index] = controlSpecification->IsInSpecificationSet(state);
		}
	}


	// Computes the winning set for which the controller currently is able to adhere to the control specification
	void Verifier::ComputeWinningSet() {
		// Initialize the winning set such that only the target set is in the winning set
		InitializeWinningSet();
		
		// Perform fixed algorithm operator on winning set to determine the winning set
		size_t iterations = 0;

		auto controlSpecificationType = _abstraction->GetControlSpecification()->GetSpecificationType();
		switch (controlSpecificationType) {
			case ControlSpecificationType::Invariance:
				iterations += PerformFixedPointAlgorithm(ControlSpecificationType::Invariance);
				break;
			case ControlSpecificationType::Reachability:
				iterations += PerformFixedPointAlgorithm(ControlSpecificationType::Reachability);
				break;
			case ControlSpecificationType::ReachAndStay:
				iterations += PerformFixedPointAlgorithm(ControlSpecificationType::Invariance); // First do an invariance fixed point algorithm 
				iterations += PerformFixedPointAlgorithm(ControlSpecificationType::Reachability); // Then do a reachability fixed point algorithm on the resulting invariance winning set 
				break;
		}

		DetermineLosingSet();

		if(_verboseMode) std::cout << std::endl << "\tFixed point iterations: " << iterations << std::endl;

		_winningSetPercentage = (float)GetWinningSetSize() / (float)_abstraction->GetStateQuantizer()->GetCardinality() * 100;
	}


	// Computes the apparant winning set
	void Verifier::ComputeApparentWinningSet() {
		std::cout << "\t\t";

		const auto spaceCardinality = _abstraction->GetStateQuantizer()->GetCardinality();
		const auto controlSpecificationType = _abstraction->GetControlSpecification()->GetSpecificationType();
		_apparentWinningCells = 0;
		
		for (unsigned long index = 0; index < spaceCardinality; index++) {
			if (index % ((long)floor(spaceCardinality / 100)) == 0) std::cout << '.';

			Vector state = _abstraction->GetStateQuantizer()->GetVectorFromIndex(index);
			bool stopEpisode = false;
			for (unsigned int i = 0; i < _maxEpisodeHorizon; i++) {
				auto stateIndex = _abstraction->GetStateQuantizer()->GetIndexFromVector(state);
				if (stateIndex == -1) {
					stopEpisode = true;
					break;
				}

				_abstraction->GetPlant()->SetState(state);

				auto controlAction = _abstraction->GetController()->GetControlActionFromIndex(stateIndex);
				state = _abstraction->GetPlant()->EvaluateDynamics(controlAction);

				switch(controlSpecificationType) {
				case ControlSpecificationType::Reachability:
					if (_abstraction->GetControlSpecification()->IsInSpecificationSet(state)) {
						_apparentWinningCells++;
						stopEpisode = true;
					}
					break;
				case ControlSpecificationType::Invariance:
					if (!_abstraction->GetControlSpecification()->IsInSpecificationSet(state)) stopEpisode = true;
					break;
				}

				if (stopEpisode) break;
			}

			if (!stopEpisode && _abstraction->GetControlSpecification()->IsInSpecificationSet(state) && controlSpecificationType == ControlSpecificationType::Invariance) _apparentWinningCells++;
		}

		_apparentWinningSetPercentage = ((float)_apparentWinningCells / (float)spaceCardinality) * 100.0;
		std::cout << std::endl;
	}


	// Performs a single fixed point iteration
	bool Verifier::PerformSingleFixedPointOperation(ControlSpecificationType type) {
		bool hasSetChanged = false;

		for (long index = 0; index < _abstraction->GetStateQuantizer()->GetCardinality(); index++) {
			auto input = _abstraction->GetController()->GetControlActionFromIndex(index);
			auto inputIndex = _abstraction->GetInputQuantizer()->GetIndexFromVector(input);

			auto ends = _abstraction->GetTransitionOfIndex(index)->GetEnds(inputIndex);

			// Determine if the transition always ends in the winning set
			bool allEndsInWinningSet = true;
			if (ends.size() == 0) allEndsInWinningSet = false;

			for (auto end : ends) {
				if (end == -1 || !_winningSet[end]) allEndsInWinningSet = false;
			}

			// Handle transitions based on specification
			switch (type) {
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


	// Performs a fixed point algorithm on the winning set based on the type
	size_t Verifier::PerformFixedPointAlgorithm(ControlSpecificationType type) {
		bool hasIterationConverged = false;
		unsigned int iterations = 0;
		while (!hasIterationConverged) {
			iterations++;
			std::cout << "\t\ti: " << iterations;

			auto hasSetChanged = PerformSingleFixedPointOperation(type);

			if (!hasSetChanged) hasIterationConverged = true;

			std::cout << std::endl;
		}
		return iterations;
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


	// Sets the episode horizon for the verifier
	void Verifier::SetEpisodeHorizon(unsigned int episodeHorizon) {
		episodeHorizon = episodeHorizon;
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
	float Verifier::GetWinningSetPercentage() {
		return _winningSetPercentage;
	}


	// Returns the apparent winning set percentage
	float Verifier::GetApparentWinningSetPercentage() {
		return _apparentWinningSetPercentage;
	}


	// Returns the percentage of completeness of the current (partial) abstraction
	float Verifier::GetAbstractionCompleteness() {
		_abstractionCompleteness = ((float)_transitionsInAbstraction / (float)_transitionsInFullAbstraction) * 100.0;
		return _abstractionCompleteness;
	}

	#pragma endregion Getters and Setters
}