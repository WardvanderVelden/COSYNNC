#pragma once

#include <thread>
#include "Abstraction.h"

namespace COSYNNC {
	class Verifier {
	public:
		// Constructor that setup up the verifier for use
		Verifier(Abstraction* abstraction);

		// Destructor
		~Verifier();

		// Computes the transition function that transitions any state in the state space to a set of states in the state space based on the control law
		void ComputeTransitions();

		// Computes a subset of the total amount of required transitions, is used for multithreading
		void ComputeSubsetOfTransitions(unsigned int threadNumber, unsigned long start, unsigned long end);

		// Computes the winning set for which the controller currently is able to adhere to the control specification
		void ComputeWinningSet();

		// Computes the apparant winning set
		void ComputeApparentWinningSet();

		// Returns whether or not an index is in the winning domain
		bool IsIndexInWinningSet(unsigned long index);

		#pragma region Getters and Setters

		// Sets a part of the winning domain, returns whether or not that element has changed
		bool SetWinningDomain(long index, bool value);

		// Sets the verbose mode
		void SetVerboseMode(bool verboseMode);

		// Sets the episode horizon for the verifier
		void SetEpisodeHorizon(unsigned int episodeHorizon);

		// Get a random vector from the space of the losing domain
		Vector GetVectorFromLosingDomain();

		// Get a random vector from the set of losing states which neighbor winning states
		Vector GetVectorFromLosingNeighborDomain();

		// Returns the last calculated percentage of the winning domain compared to the state space
		float GetWinningSetPercentage();

		// Returns the apparent winning set percentage
		float GetApparentWinningSetPercentage();

		// Returns the percentage of completeness of the current (partial) abstraction
		float GetAbstractionCompleteness();

		// Returns the size of the winning set compared to the cardinality of the state space
		long GetWinningSetSize();

		#pragma endregion Getters and Setters
	private:
		// Performs a single fixed point iteration, returns whether or not the set has changed
		bool PerformSingleFixedPointOperation(ControlSpecificationType type);

		// Performs a fixed point algorithm on the winning set based on the type
		size_t PerformFixedPointAlgorithm(ControlSpecificationType type);

		// Determines the losing set and the set of losing cells which are next to the winning domain
		void DetermineLosingSet();

		// Initializes the winning set for verification
		void InitializeWinningSet();


		Abstraction* _abstraction;

		unsigned long _transitionsInFullAbstraction;
		unsigned long _transitionsInAbstraction;
		float _abstractionCompleteness = 0.0;

		unsigned int _maxEpisodeHorizon = 50;
		unsigned long _apparentWinningCells = 0;

		bool* _winningSet;

		vector<long> _losingIndices;
		vector<long> _losingWinningNeighborIndices;

		float _winningSetPercentage = 0.0;
		float _apparentWinningSetPercentage = 0.0;

		bool _verboseMode = false;
	};
}