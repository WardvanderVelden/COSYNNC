#pragma once
#include "Plant.h"
#include "Controller.h"
#include "Quantizer.h"
#include "Transition.h"
#include "Hyperplane.h"

namespace COSYNNC {
	class Verifier {
	public:
		Verifier(Plant* plant, Controller* controller, Quantizer* stateQuantizer, Quantizer* inputQuantizer);

		~Verifier();

		// Computes the transition function that transitions any state in the state space to a set of states in the state space based on the control law
		void ComputeTransitionFunction();

		// Computes the transition function for a single index
		void ComputeTransitionFunctionForIndex(long index, Vector input);

		// Computes the winning set for which the controller currently is able to adhere to the control specification
		void ComputeWinningSet();

		// Determines the losing set and the set of losing cells which are next to the winning domain
		void DetermineLosingSet();

		//  Prints a verbose walk of the current controller using greedy inputs
		void PrintVerboseWalk(Vector initialState);

		// Returns the size of the winning set compared to the cardinality of the state space
		long GetWinningSetSize();

		// Sets a part of the winning domain, returns whether or not that element has changed
		bool SetWinningDomain(long index, bool value);

		// Sets the verbose mode
		void SetVerboseMode(bool verboseMode);

		// Get a random vector from the space of the losing domain
		Vector GetVectorFromLosingDomain();

		// Get a random vector from the set of losing states which neighbor winning states
		Vector GetVectorFromLosingNeighborDomain();

		// Over-approximates the vertices of the original cell and returns the hyperplanes that result from the over-approximation
		Vector* OverApproximateEvolution(Vector state, Vector input, vector<Hyperplane>& hyperplanes);

		// Returns the hyperplanes that naturally arise between the vertices
		vector<Hyperplane> GetHyperplanesBetweenVertices(Vector* vertices, Vector cellCenter);

		// Flood fills between planes, adding the indices of the cells to the transitions of the origin cell
		void FloodfillBetweenHyperplanes(unsigned long index, Vector center, vector<Hyperplane>& planes);

		// Generates the appropriate floodfill indices based on the current inex and the processed indices
		void AddFloodfillOrder(Vector center, Vector direction, vector<unsigned long>& indices, vector<unsigned long>& processedIndices);

		// Checks if a point is contained between planes
		bool IsPointBetweenHyperplanes(Vector point, vector<Hyperplane>& planes);

		// LEGACY: Returns the edges between a set of vertices if the vertices are properly sorted
		Edge* GetEdgesBetweenVertices(Vector* vertices);

		// Returns the last calculated percentage of the winning domain compared to the state space
		float GetWinningDomainPercentage();

		// Returns whether or not an index is in the winning domain
		bool IsIndexInWinningSet(unsigned long index);
	private:
		Plant* _plant;
		Controller* _controller;
		Quantizer* _stateQuantizer;
		Quantizer* _inputQuantizer;
		ControlSpecification* _specification;

		Transition* _transitions;
		bool* _winningSet;

		vector<long> _losingIndices;
		vector<long> _losingWinningNeighborIndices;

		float _winningDomainPercentage = 0.0;

		const unsigned int _maxSteps = 50;

		const float _interpolationPrecisionFactor = 0.1;

		unsigned int _spaceDimension;
		unsigned long _spaceCardinality;
		Vector _spaceEta;

		unsigned int _inputDimension;
		unsigned long _inputCardinality;

		unsigned int _amountOfVerticesPerCell;
		unsigned int _amountOfEdgesPerCell;

		bool _verboseMode = false;
	};
}