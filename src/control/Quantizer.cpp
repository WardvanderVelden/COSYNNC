#include "Quantizer.h"

namespace COSYNNC {
	Quantizer::Quantizer() { }


	// Set the quantization parameters for the space, if the space is not bounded the lower bound will be used as the reference
	void Quantizer::SetQuantizeParameters(Vector spaceEta, Vector spaceLowerBound, Vector spaceUpperBound) {
		_spaceDimension = spaceEta.GetLength();
		_spaceEta = spaceEta;

		_spaceLowerVertex = Vector(_spaceDimension);
		_spaceUpperVertex = Vector(_spaceDimension);

		_spaceCardinalityPerAxis = vector<unsigned long>(_spaceDimension, 0);
		_indicesPerDimension = vector<unsigned long>(_spaceDimension, 0);

		for (size_t dim = 0; dim < _spaceDimension; dim++) {
			auto lowerIndex = ceil(spaceLowerBound[dim] / _spaceEta[dim]);
			_spaceLowerVertex[dim] = lowerIndex * _spaceEta[dim];

			auto upperIndex = floor(spaceUpperBound[dim] / _spaceEta[dim]);
			_spaceUpperVertex[dim] = upperIndex * _spaceEta[dim];
			
			_spaceCardinalityPerAxis[dim] = (unsigned long)abs(upperIndex - lowerIndex) + 1;

			if (dim == 0) {
				_spaceCardinality = _spaceCardinalityPerAxis[dim];
				_indicesPerDimension[dim] = 1;
			}
			else {
				_spaceCardinality *= _spaceCardinalityPerAxis[dim];
				_indicesPerDimension[dim] = _indicesPerDimension[dim - 1] * _spaceCardinalityPerAxis[dim - 1];
			}
		}
	}


	// Quantize a vector to quantization parameters
	Vector Quantizer::QuantizeVector(Vector vector) {
		Vector quantized(_spaceDimension);

		for (size_t dim = 0; dim < _spaceDimension; dim++) {
			if (vector[dim] < _spaceLowerVertex[dim]) vector[dim] = _spaceLowerVertex[dim];
			if (vector[dim] > _spaceUpperVertex[dim]) vector[dim] = _spaceUpperVertex[dim];

			quantized[dim] = round(vector[dim] / _spaceEta[dim]) * _spaceEta[dim];
		}

		return quantized;
	}


	// Quantizes a normalized vector to a normalized quantized vector
	Vector Quantizer::QuantizeNormalizedVector(Vector vector) {
		auto denormalized = DenormalizeVector(vector);
		auto quantized = QuantizeVector(denormalized);

		return NormalizeVector(quantized);
	}


	// Normalize a vector from the bounded space to the normal space
	Vector Quantizer::NormalizeVector(Vector denormal) {
		Vector normal(_spaceDimension);

		for (size_t dim = 0; dim < _spaceDimension; dim++) {
			if (denormal[dim] < _spaceLowerVertex[dim]) denormal[dim] = _spaceLowerVertex[dim];
			if (denormal[dim] > _spaceUpperVertex[dim]) denormal[dim] = _spaceUpperVertex[dim];

			normal[dim] = (denormal[dim] - _spaceLowerVertex[dim]) / (_spaceUpperVertex[dim] - _spaceLowerVertex[dim]);
		}

		return normal;
	}


	// Denormalize a vector a vector from the normal space to the bounded space
	Vector Quantizer::DenormalizeVector(Vector normal) {
		Vector denormal(_spaceDimension);

		for (size_t dim = 0; dim < _spaceDimension; dim++) {
			denormal[dim] = normal[dim] * (_spaceUpperVertex[dim] - _spaceLowerVertex[dim]) + _spaceLowerVertex[dim];
		}

		return denormal;
	}


	// Checks if a vector is in the bounds of the quantized space
	bool Quantizer::IsInBounds(Vector vector) {
		for (int dim = 0; dim < _spaceDimension; dim++) {
			if (vector[dim] < _spaceLowerVertex[dim] || vector[dim] > _spaceUpperVertex[dim]) {
				return false;
			}
		}

		return true;
	}


	#pragma region Getters

	// Returns the input that corresponds to the labelled output of the network
	Vector Quantizer::GetVectorFromOneHot(Vector oneHot) {
		int index = 0;
		for (size_t i = 0; i < oneHot.GetLength(); i++) {
			if (oneHot[i] == 1.0) index = i;
		}

		return GetVectorFromIndex(index);
	}


	// Returns the vector that corresponds to the index in the space
	Vector Quantizer::GetVectorFromIndex(long index) {
		Vector vec(_spaceDimension);
		for (int dim = (_spaceDimension - 1); dim >= 0; dim--) {
			long indexOnAxis = floor(index / _indicesPerDimension[dim]);
			vec[dim] = indexOnAxis * _spaceEta[dim] + _spaceLowerVertex[dim];

			index -= indexOnAxis * _indicesPerDimension[dim];
		}

		return vec;
	}


	// Returns the index that corresponds to that vector
	unsigned long Quantizer::GetIndexFromVector(Vector vector) {
		unsigned long index = 0;

		if (!IsInBounds(vector)) return -1;

		for (size_t dim = 0; dim < _spaceDimension; dim++) {
			index += round((vector[dim] - _spaceLowerVertex[dim]) / _spaceEta[dim]) * _indicesPerDimension[dim];
		}

		return index;
	}


	// Returns the axis indices 
	vector<unsigned long> Quantizer::GetAxisIndicesFromIndex(unsigned long index) {
		vector<unsigned long> axisIndices = vector<unsigned long>(_spaceDimension, 0);

		for (int dim = (_spaceDimension - 1); dim >= 0; dim--) {
			long indexOnAxis = floor(index / _indicesPerDimension[dim]);
			axisIndices[dim] = indexOnAxis;

			index -= indexOnAxis * _indicesPerDimension[dim];
		}

		return axisIndices;
	}


	// Returns the indices on every axis from a vector
	vector<unsigned long> Quantizer::GetAxisIndicesFromVector(Vector vector) {
		auto index = GetIndexFromVector(vector);
		return GetAxisIndicesFromIndex(index);
	}


	// Returns the global index form a vector of axis indices
	unsigned long Quantizer::GetIndexFromAxisIndices(vector<unsigned long> axisIndices) {
		unsigned long index = 0;
		for (size_t i = 0; i < _spaceDimension; i++) index += axisIndices[i] * _indicesPerDimension[i];

		return index;
	}


	// Gets a random vector within the bounded quantization space if isBounded is set
	Vector Quantizer::GetRandomVector() {
		Vector randomVector(_spaceDimension);

		for (int i = 0; i < _spaceDimension; i++) {
			float randomFloat = (float)rand() / RAND_MAX;
			randomVector[i] = _spaceLowerVertex[i] + (_spaceUpperVertex[i] - _spaceLowerVertex[i]) * randomFloat;
		}

		return QuantizeVector(randomVector);
	}


	// Returns the cardinality of the quantized set
	long Quantizer::GetCardinality() const {
		return _spaceCardinality;
	}


	// Returns the dimension of the space
	int Quantizer::GetDimension() const {
		return _spaceDimension;
	}


	// Returns the lower bound of the quantizer
	Vector Quantizer::GetLowerBound() const {
		return _spaceLowerVertex;
	}


	// Returns the upper bound of the quantizer
	Vector Quantizer::GetUpperBound() const {
		return _spaceUpperVertex;
	}


	// Returns the space eta of the quantizer
	Vector Quantizer::GetEta() const {
		return _spaceEta;
	}


	// Returns an array of vectors which are the vertices of the hyper cell
	Vector* Quantizer::GetCellVertices(Vector cell) {
		const unsigned int amountOfVertices = pow(2.0, (double)_spaceDimension);

		auto cellCenter = QuantizeVector(cell);
		Vector* vertices = new Vector[amountOfVertices];

		auto baseVertex = cellCenter;
		for (unsigned int i = 0; i < _spaceDimension; i++) baseVertex[i] -= _spaceEta[i] * 0.5;

		unsigned int vertexIndex = 0;
		for (unsigned int i = 0; i < _spaceDimension; i++) {
			if (i == 0) vertices[vertexIndex++] = baseVertex;

			auto verticesAllocated = vertexIndex;
			for (unsigned int j = 0; j < verticesAllocated; j++) {
				auto facingVertex = vertices[j];

				auto newVertex = facingVertex;
				newVertex[i] += _spaceEta[i];
				vertices[vertexIndex++] = newVertex;
			}
		}

		return vertices;
	}


	// Returns an array of vectors which are the vertices of the hyper cell
	Vector* Quantizer::GetCellVertices(unsigned long cellIndex) {
		const unsigned int amountOfVertices = pow(2.0, (double)_spaceDimension);

		auto cellCenter = GetVectorFromIndex(cellIndex);
		Vector* vertices = new Vector[amountOfVertices];

		auto baseVertex = cellCenter;
		for (unsigned int i = 0; i < _spaceDimension; i++) baseVertex[i] -= _spaceEta[i] * 0.5;

		unsigned int vertexIndex = 0;
		for (unsigned int i = 0; i < _spaceDimension; i++) {
			if (i == 0) vertices[vertexIndex++] = baseVertex;

			auto verticesAllocated = vertexIndex;
			for (unsigned int j = 0; j < verticesAllocated; j++) {
				auto facingVertex = vertices[j];

				auto newVertex = facingVertex;
				newVertex[i] += _spaceEta[i];
				vertices[vertexIndex++] = newVertex;
			}
		}

		return vertices;
	}

	#pragma endregion Getters
}