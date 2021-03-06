#pragma once

#include "Plant.h"
#include "Controller.h"
#include "Quantizer.h"
#include "Transition.h"
#include "Hyperplane.h"

using namespace std;

namespace COSYNNC {
	class Abstraction {
	public:
		// Default constructor
		Abstraction();

		// Constructor that fully defines the abstraction
		Abstraction(Plant* plant, Controller* controller, Quantizer* stateQuantizer, Quantizer* inputQuantizer, ControlSpecification* controlSpecification);

		// Destructor
		~Abstraction();

		// Getters
		#pragma region Getters

		Plant* GetPlant() const { return _plant; }
		Controller* GetController() const { return _controller; }
		Quantizer* GetStateQuantizer() const { return _stateQuantizer; }
		Quantizer* GetInputQuantizer() const { return _inputQuantizer; }
		ControlSpecification* GetControlSpecification() const { return _controlSpecification; }

		#pragma endregion Getters

		// Computes the transition function for a single index
		bool ComputeTransitionFunctionForIndex(long index, Vector input);

		// Returns a reference to the transition based on the index
		Transition* GetTransitionOfIndex(unsigned long index);

		// Returns the amount of transitions that the abstraction contains
		unsigned long GetAmountOfTransitions() const { return _amountOfTransitions; };

		unsigned long GetAmountOfEnds() const { return _amountOfEnds; }

		// Sets whether or not to use the rough transition scheme
		void SetUseRefinedTransitions(bool use = false);

		// Sets whether or not to save the transitions
		void SetSaveTransitions(bool save = true);

		// Returns whether or not the abstraction is made up of refined transitions
		bool IsUsingRefinedTransitions() const { return _useRefinedTransitions; };

		// Returns whether or not the transitions are being saved
		bool IsSavingTransitions() const { return _saveTransitions; };

		// Empties the abstraction to save data
		void EmptyTransitions();
	private:
		#pragma region Transition Function

		// Over-approximates the vertices of the original cell and returns the hyperplanes that result from the over-approximation
		Vector* OverApproximateEvolution(Vector state, Vector newState, Vector input, vector<Hyperplane>& hyperplanes);

		// Returns the hyperplanes that naturally arise between the vertices
		vector<Hyperplane> GetHyperplanesBetweenVertices(Vector* vertices, Vector cellCenter);

		// Flood fills between planes, adding the indices of the cells to the transitions of the origin cell
		void FloodfillBetweenHyperplanes(unsigned long index, Vector center, vector<Hyperplane>& planes, unsigned long inputIndex);

		// Fills the hyper rectangle formed by the upper and lower bound of the vertices
		void FillHyperRectangleBetweenBounds(unsigned long index, Transition* transition, unsigned long inputIndex);

		// Find the upper and lower bound of the transition and set it
		void ComputeTransitionBounds(Transition* transition, Vector* vertices, Vector post, unsigned long inputIndex);

		// Generates the appropriate floodfill indices based on the current inex and the processed indices
		void AddFloodfillOrder(Vector center, Vector direction, vector<unsigned long>& indices, vector<unsigned long>& processedIndices);

		// Checks if a point is contained between planes
		bool IsPointBetweenHyperplanes(Vector point, vector<Hyperplane>& planes);

		// Calculates the vertices to hyperplane distribution
		void CalculateVerticesOnHyperplaneDistribution();

		#pragma endregion Transition Function

		// Abstraction settings
		bool _useRefinedTransitions = true;
		bool _saveTransitions = true;

		// Abstraction components
		Plant* _plant = nullptr;
		Controller* _controller = nullptr;
		Quantizer* _stateQuantizer = nullptr;
		Quantizer* _inputQuantizer = nullptr;
		ControlSpecification* _controlSpecification = nullptr;

		// Transition variables
		unsigned int _partitions = 1; // thread::hardware_concurrency();
		unsigned long _partitionSize = 0;
		Transition** _transitionPartitions = new Transition*[_partitions];

		unsigned long _amountOfTransitions = 0;
		unsigned long _amountOfEnds = 0;

		unsigned int _amountOfVerticesPerCell = 0;
		unsigned int _amountOfEdgesPerCell = 0;

		vector<vector<short>> _radialGrowthDistribution;
		vector<vector<unsigned short>> _verticesOnHyperplaneDistribution;
		vector<Vector> _normalsOfHyperplane;

		bool _transitionsAreEmpty = true;
	};
}