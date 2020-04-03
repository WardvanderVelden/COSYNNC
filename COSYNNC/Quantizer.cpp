#include "Quantizer.h"

namespace COSYNNC {
	Quantizer::Quantizer(bool isBounded) {
		_isBounded = isBounded;
	}


	// Set the quantization parameters for the space
	void Quantizer::SetQuantizeParameters(Vector spaceEta, Vector spaceReference) {
		_spaceDimension = spaceEta.GetLength();
		_spaceEta = spaceEta;

		_spaceReference = spaceReference;

		_isBounded = false;
	}


	// Set the quantization parameters for the space, if the space is not bounded the lower bound will be used as the reference
	void Quantizer::SetQuantizeParameters(Vector spaceEta, Vector spaceLowerBound, Vector spaceUpperBound) {
		_spaceDimension = spaceEta.GetLength();
		_spaceEta = spaceEta;

		if (_isBounded) {
			_spaceLowerBound = spaceLowerBound;
			_spaceUpperBound = spaceUpperBound;

			_spaceCardinalityPerAxis = vector<long>(_spaceDimension, 0);

			for (int i = 0; i < _spaceDimension; i++) {
				if (i == 0) _spaceCardinality = round((_spaceUpperBound[i] - _spaceLowerBound[i]) / _spaceEta[i]);
				else _spaceCardinality *= round((_spaceUpperBound[i] - _spaceLowerBound[i]) / _spaceEta[i]);

				_spaceCardinalityPerAxis[i] = round((_spaceUpperBound[i] - _spaceLowerBound[i]) / _spaceEta[i]);
			}
		}
		else {
			_spaceReference = spaceLowerBound;
		}
	}


	// Quantize a vector to quantization parameters
	Vector Quantizer::QuantizeVector(Vector vector) {
		Vector quantized(vector.GetLength());

		for (int i = 0; i < vector.GetLength(); i++) {
			if (_isBounded) {
				if (vector[i] < _spaceLowerBound[i]) vector[i] = _spaceLowerBound[i] + _spaceEta[i] * 0.5;
				if (vector[i] > _spaceUpperBound[i]) vector[i] = _spaceUpperBound[i] - _spaceEta[i] * 0.5;

				quantized[i] = floor(((vector[i] - _spaceLowerBound[i]) / _spaceEta[i])) * _spaceEta[i] + _spaceLowerBound[i] + _spaceEta[i] * 0.5;
			}
			else {
				quantized[i] = floor(((vector[i] - _spaceReference[i]) / _spaceEta[i])) * _spaceEta[i] + _spaceReference[i] + _spaceEta[i] * 0.5;
			}
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
		Vector normal = Vector(denormal.GetLength());

		for (int i = 0; i < denormal.GetLength(); i++) {
			// Project back to the bounds
			if (denormal[i] > (_spaceUpperBound[i] - _spaceEta[i] * 0.5)) denormal[i] = _spaceUpperBound[i] - _spaceEta[i] * 0.5;
			if (denormal[i] < (_spaceLowerBound[i] + _spaceEta[i] * 0.5)) denormal[i] = _spaceLowerBound[i] + _spaceEta[i] * 0.5;

			// Normalize
			normal[i] = (denormal[i] - _spaceLowerBound[i]) / (_spaceUpperBound[i] - _spaceLowerBound[i]);
		}

		return normal;
	}


	// Denormalize a vector a vector from the normal space to the bounded space
	Vector Quantizer::DenormalizeVector(Vector normal) {
		Vector denormal = Vector(normal.GetLength());

		for (int i = 0; i < normal.GetLength(); i++) {
			denormal[i] = normal[i] * (_spaceUpperBound[i] - _spaceLowerBound[i]) + _spaceLowerBound[i];
		}

		return denormal;
	}


	// Checks if a vector is in the bounds of the quantized space
	bool Quantizer::IsInBounds(Vector vector) {
		for (int i = 0; i < vector.GetLength(); i++) {
			if (vector[i] < _spaceLowerBound[i] || vector[i] > _spaceUpperBound[i]) {
				return false;
			}
		}

		return true;
	}


	#pragma region Getters

	// Returns the input that corresponds to the labelled output of the network
	Vector Quantizer::GetVectorFromOneHot(Vector oneHot) {
		int index = 0;
		for (int i = 0; i < oneHot.GetLength(); i++) {
			if (oneHot[i] == 1.0) index = i;
		}

		return GetVectorFromIndex(index);
	}


	// Returns the vector that corresponds to the index in the space
	Vector Quantizer::GetVectorFromIndex(long index) {
		Vector vec(_spaceDimension);
		for (int i = (_spaceDimension - 1); i >= 0; i--) {
			long indexOnAxis = (i > 0) ? floor(index / _spaceCardinalityPerAxis[i - 1]) : index;

			if (i > 0) index -= indexOnAxis * _spaceCardinalityPerAxis[i - 1];
			vec[i] = indexOnAxis * _spaceEta[i] + _spaceLowerBound[i] + _spaceEta[i] * 0.5;
		}

		return vec;
	}


	// Returns the index that corresponds to that vector
	long Quantizer::GetIndexFromVector(Vector vector) {
		long index = 0;

		if (!IsInBounds(vector)) return -1;

		for (int i = 0; i < _spaceDimension; i++) {
			long indexPerAxis = (i != 0) ? _spaceCardinalityPerAxis[i - 1] : 1;

			index += floor((vector[i] - _spaceLowerBound[i]) / _spaceEta[i]) * indexPerAxis;
		}

		return index;
	}


	// Gets a random vector within the bounded quantization space if isBounded is set
	Vector Quantizer::GetRandomVector() {
		Vector randomVector(_spaceDimension);

		if (_isBounded) {
			for (int i = 0; i < _spaceDimension; i++) {
				float randomFloat = (float)rand() / RAND_MAX;
				randomVector[i] = _spaceLowerBound[i] + (_spaceUpperBound[i] - _spaceLowerBound[i]) * randomFloat;
			}
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
		return _spaceLowerBound;
	}


	// Returns the upper bound of the quantizer
	Vector Quantizer::GetUpperBound() const {
		return _spaceUpperBound;
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