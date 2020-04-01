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

		Transition* GetTransitionOfIndex(unsigned long index) const { return &_transitions[index]; }

		#pragma endregion Getters

		// Computes the transition function for a single index
		void ComputeTransitionFunctionForIndex(long index, Vector input);
	private:
		#pragma region Transition Function

		// Over-approximates the vertices of the original cell and returns the hyperplanes that result from the over-approximation
		Vector* OverApproximateEvolution(Vector state, Vector input, vector<Hyperplane>& hyperplanes);

		// Returns the hyperplanes that naturally arise between the vertices
		vector<Hyperplane> GetHyperplanesBetweenVertices(Vector* vertices, Vector cellCenter);

		// Flood fills between planes, adding the indices of the cells to the transitions of the origin cell
		void FloodfillBetweenHyperplanes(unsigned long index, Vector center, vector<Hyperplane>& planes, unsigned long inputIndex);

		// Generates the appropriate floodfill indices based on the current inex and the processed indices
		void AddFloodfillOrder(Vector center, Vector direction, vector<unsigned long>& indices, vector<unsigned long>& processedIndices);

		// Checks if a point is contained between planes
		bool IsPointBetweenHyperplanes(Vector point, vector<Hyperplane>& planes);

		// Calculates the vertices to hyperplane distribution
		void CalculateVerticesOnHyperplaneDistribution();

		#pragma endregion Transition Function

		// Abstraction components
		Plant* _plant = nullptr;
		Controller* _controller = nullptr;
		Quantizer* _stateQuantizer = nullptr;
		Quantizer* _inputQuantizer = nullptr;
		ControlSpecification* _controlSpecification = nullptr;

		// Transition variables
		Transition* _transitions;

		unsigned int _amountOfVerticesPerCell = 0;
		unsigned int _amountOfEdgesPerCell = 0;

		vector<vector<unsigned short>> _verticesOnHyperplaneDistribution;
		vector<Vector> _normalsOfHyperplane;
	};
}