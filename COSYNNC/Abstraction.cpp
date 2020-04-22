#pragma once

#include "Abstraction.h"

using namespace std;

namespace COSYNNC {
	// Default constructor
	Abstraction::Abstraction() { }


	// Constructor that fully defines the abstraction
	Abstraction::Abstraction(Plant* plant, Controller* controller, Quantizer* stateQuantizer, Quantizer* inputQuantizer, ControlSpecification* controlSpecification) {
		// Set references to abstraction components
		_plant = plant;
		_controller = controller;

		_stateQuantizer = stateQuantizer;
		_inputQuantizer = inputQuantizer;

		_controlSpecification = controlSpecification;

		// Initialize soft constants for transition calculations
		_amountOfVerticesPerCell = pow(2.0, (double)_stateQuantizer->GetDimension());
		_amountOfEdgesPerCell = _stateQuantizer->GetDimension() * pow(2.0, (double)_stateQuantizer->GetDimension() - 1);

		CalculateVerticesOnHyperplaneDistribution();

		// Initialize transitions
		auto spaceCardinality = _stateQuantizer->GetCardinality();

		/*_transitions = new Transition[spaceCardinality];
		for (unsigned long index = 0; index < spaceCardinality; index++) {
			_transitions[index] = Transition(index, _inputQuantizer->GetCardinality());
		}*/

		_partitionSize = floor((float)spaceCardinality / (float)_partitions);
		for (unsigned int i = 0; i < _partitions; i++) {
			auto size = _partitionSize;
			if (i == (_partitions - 1)) {
				size = spaceCardinality -  (i * _partitionSize);
			}

			_transitionPartitions[i] = new Transition[size];
			for (unsigned long j = 0; j < size; j++) {
				_transitionPartitions[i][j] = Transition(_partitionSize * i + j, _inputQuantizer->GetCardinality());
			}
		}
	}


	// Destructor
	Abstraction::~Abstraction() {
		//delete[] _transitions;

		for (unsigned int i = 0; i < _partitions; i++) delete[] _transitionPartitions[i];
		delete[] _transitionPartitions;
	}


	// Returns a reference to the transition based on the index
	Transition* Abstraction::GetTransitionOfIndex(unsigned long index) {
		unsigned int partition = floor(index / _partitionSize);
		unsigned long partitionIndex = index - partition * _partitionSize;

		return &_transitionPartitions[partition][partitionIndex];
	}


	#pragma region Transition Functions

	// Computes the transition function for a single index, returns true if any calculations were made
	bool Abstraction::ComputeTransitionFunctionForIndex(long index, Vector input) {
		auto inputIndex = _inputQuantizer->GetIndexFromVector(input);
		auto transition = GetTransitionOfIndex(index);
		if (!transition->HasTransitionBeenCalculated(inputIndex)) {
			auto state = _stateQuantizer->GetVectorFromIndex(index);
			_plant->SetState(state);

			auto newState = _plant->EvaluateOverApproximation(input);

			// Check if newState is in bounds of the state space, if not then we know the transition already
			if (!_stateQuantizer->IsInBounds(newState)) {
				transition->AddEnd(-1, inputIndex);
				transition->SetInputProcessed(inputIndex);
				return true;
			}

			// Evolve the vertices of the cell and find the resulting hyperplanes
			vector<Hyperplane> hyperplanes;
			auto vertices = OverApproximateEvolution(state, input, hyperplanes);

			// Check if any of the vertices is out of bounds, in that case the transition is invalid
			for (unsigned int i = 0; i < _amountOfVerticesPerCell; i++) {
				if (!_stateQuantizer->IsInBounds(vertices[i])) {
					transition->AddEnd(-1, inputIndex);
					transition->SetInputProcessed(inputIndex);
					return true;
				}
			}

			// Get the point in space that represents the center of the vertices
			auto center = Vector(_stateQuantizer->GetDimension());
			for (unsigned int i = 0; i < _amountOfVerticesPerCell; i++) {
				center = center + vertices[i];
			}
			center = center * (1.0 / (float)_amountOfVerticesPerCell);

			// Flood fill as long as a vertex in a cell is in the internal area
			FloodfillBetweenHyperplanes(index, center, hyperplanes, inputIndex);

			// Determine the lower and upper bounds of the transition and add these to the transition so that SCOTS can work with them if that is set
			if(_doCalculateLowerAndUpperBound) SetTransitionPostAndBounds(transition, vertices, newState, inputIndex);

			// Set the input as processed
			transition->SetInputProcessed(inputIndex);

			// Free up memory
			delete[] vertices;

			return true;
		}
		return false;
	}


	// Over approximates all the vertices and returns an array of the new vertices
	Vector* Abstraction::OverApproximateEvolution(Vector state, Vector input, vector<Hyperplane>& hyperplanes) {
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


	// Find the upper and lower bound of the transition and set it
	void Abstraction::SetTransitionPostAndBounds(Transition* transition, Vector* vertices, Vector post, unsigned long inputIndex) {
		Vector lowerBound = post; 
		Vector upperBound = post;

		for (unsigned int i = 0; i < _amountOfVerticesPerCell; i++) {
			auto vertex = vertices[i];
			for (size_t j = 0; j < _stateQuantizer->GetDimension(); j++) {
				lowerBound[j] = min(lowerBound[j], vertex[j]);
				lowerBound[j] = min(lowerBound[j], vertex[j]);
			}
		}

		transition->SetPost(post, inputIndex);
		transition->SetLowerAndUpperBound(lowerBound, upperBound, inputIndex);
	}


	// Returns the hyperplanes that naturally arise between the vertices
	vector<Hyperplane> Abstraction::GetHyperplanesBetweenVertices(Vector* vertices, Vector cellCenter) {
		vector<Hyperplane> hyperplanes;

		for (unsigned int i = 0; i < (_stateQuantizer->GetDimension() * 2); i++) {
			Hyperplane hyperplane(_stateQuantizer->GetDimension());

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
	void Abstraction::FloodfillBetweenHyperplanes(unsigned long index, Vector center, vector<Hyperplane>& planes, unsigned long inputIndex) {
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

			if (isBetweenPlanes || currentIndex == centerIndex) {
				// Add order to transitions
				GetTransitionOfIndex(index)->AddEnd(currentIndex, inputIndex);
				_amountOfTransitions++;

				// Generate new orders that branch from current order
				auto cellCenter = _stateQuantizer->GetVectorFromIndex(currentIndex);
				for (unsigned int i = 0; i < _stateQuantizer->GetDimension(); i++) {
					Vector direction(_stateQuantizer->GetDimension());

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
	void Abstraction::AddFloodfillOrder(Vector center, Vector direction, vector<unsigned long>& indices, vector<unsigned long>& processedIndices) {
		// Get index of neighbor cell that is in that direction
		auto neighborCell = center + (direction * _stateQuantizer->GetEta());
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
	bool Abstraction::IsPointBetweenHyperplanes(Vector point, vector<Hyperplane>& planes) {
		for (unsigned int i = 0; i < planes.size(); i++) {
			if (!planes[i].IsPointOnInternalSide(point)) return false;
		}

		return true;
	}


	// Calculates the vertices to hyperplane distribution
	void Abstraction::CalculateVerticesOnHyperplaneDistribution() {
		auto cellCenter = _stateQuantizer->GetVectorFromIndex(0);
		auto vertices = _stateQuantizer->GetCellVertices(0);

		for (unsigned int dim = 0; dim < _stateQuantizer->GetDimension(); dim++) {
			Vector normal(_stateQuantizer->GetDimension());

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

	#pragma endregion Transition Functions
};