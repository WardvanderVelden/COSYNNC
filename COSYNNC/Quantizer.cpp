#include "Quantizer.h"

using namespace COSYNNC;

Quantizer::Quantizer(bool isBounded) {
	_isBounded = isBounded;
}


// Set the quantization parameters for the space
void Quantizer::SetQuantizeParameters(Vector spaceEta, Vector spaceReference) {
	_spaceDim = spaceEta.GetLength();
	_spaceEta = spaceEta;

	_spaceReference = spaceReference;

	_isBounded = false;
}


// Set the quantization parameters for the space, if the space is not bounded the lower bound will be used as the reference
void Quantizer::SetQuantizeParameters(Vector spaceEta, Vector spaceLowerBound, Vector spaceUpperBound) {
	_spaceDim = spaceEta.GetLength();
	_spaceEta = spaceEta;
		
	if (_isBounded) {
		_spaceLowerBound = spaceLowerBound;
		_spaceUpperBound = spaceUpperBound;

		_spaceCardinalityPerAxis = vector<long>(_spaceDim, 0);

		for (int i = 0; i < _spaceDim; i++) {
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
		if (denormal[i] > (_spaceUpperBound[i] - _spaceEta[i]*0.5)) denormal[i] = _spaceUpperBound[i] - _spaceEta[i] * 0.5;
		if (denormal[i] < (_spaceLowerBound[i] + _spaceEta[i]*0.5)) denormal[i] = _spaceLowerBound[i] + _spaceEta[i] * 0.5;

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


// Returns the nearest quantized element in the quantized space with a probability and the alternative
vector<ProbabilisticVector> Quantizer::QuantizeVectorProbabilistically(Vector denormal) {
	vector<ProbabilisticVector> vectors;

	Vector quantized = QuantizeVector(denormal);

	if (!IsInBounds(denormal)) {
		vectors.push_back({ quantized, 1.0 });
		return vectors;
	}

	// Calculate sigma
	Vector sigma = quantized - denormal;
	for (int i = 0; i < sigma.GetLength(); i++) sigma[i] = 1.0 * abs(sigma[i]);

	// Define probability distribution function
	// TODO: Make it n-dimensional
	auto pdf = [](Vector x, Vector mu, Vector sigma) {
		return 1 / (sigma[0] * sqrt(2 * PI)) * exp(-0.5 * pow((x[0] - mu[0]) / sigma[0], 2));
	};

	// Calculate integral to determine probability
	float dx = 1000;

	Vector lowerVertex = denormal - (sigma * 3.0);
	Vector upperVertex = denormal + (sigma * 3.0);

	Vector currentPoint(lowerVertex);

	Vector oldNearest = NULL;
	int i = -1;
	float total = 0.0;
	while (currentPoint[0] < upperVertex[0]) {
		auto nearest = QuantizeVector(currentPoint);
		if (!(oldNearest == nearest)) {
			oldNearest = nearest;
			vectors.push_back({ nearest, 0.0 });
			i++;
		}

		float addition = pdf(currentPoint, denormal, sigma) * dx;
		vectors[i].probability += addition;
		total += addition;

		currentPoint[0] += dx;
	}

	// Normalize to compensate for numerical inaccuracy
	for (int i = 0; i < vectors.size(); i++) vectors[i].probability = vectors[i].probability / total;

	return vectors;
}


// Returns the input that corresponds to the labelled output of the network
Vector Quantizer::GetVectorFromOneHot(Vector oneHot) {
	int index = 0;
	for (int i = 0; i < oneHot.GetLength(); i++) {
		if (oneHot[i] == 1.0) index = i;
	}

	Vector vec(_spaceDim);
	for (int i = (_spaceDim - 1); i >= 0; i--) {
		int indexOnAxis = 0;
		if (i > 0) indexOnAxis = floor(index / _spaceCardinalityPerAxis[i]);
		else indexOnAxis = index;
		index -= indexOnAxis * _spaceCardinalityPerAxis[i];
		vec[i] = indexOnAxis * _spaceEta[i] + _spaceLowerBound[i] + _spaceEta[i] * 0.5;
	}

	return vec;
}


// Returns the vector that corresponds to the index in the space
Vector Quantizer::GetVectorFromIndex(long index) {
	Vector vec(_spaceDim);
	for (int i = (_spaceDim - 1); i >= 0; i--) {
		long indexOnAxis = indexOnAxis = (i != 0) ? floor(index / _spaceCardinalityPerAxis[i - 1]) : index;
		
		index -= indexOnAxis * _spaceCardinalityPerAxis[i];
		vec[i] = indexOnAxis * _spaceEta[i] + _spaceLowerBound[i] + _spaceEta[i] * 0.5;
	}

	//if (IsInBounds(vec)) return vec;
	
	return vec;
}


// Returns the index that corresponds to that vector
long Quantizer::GetIndexFromVector(Vector vector) {
	long index = 0;

	if (!IsInBounds(vector)) return -1;

	for (int i = 0; i < _spaceDim; i++) {
		long indexPerAxis = (i != 0) ? _spaceCardinalityPerAxis[i - 1] : 1;

		index += floor((vector[i] - _spaceLowerBound[i]) / _spaceEta[i]) * indexPerAxis;
	}

	return index;
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


// Gets a random vector within the bounded quantization space if isBounded is set
Vector Quantizer::GetRandomVector() {
	Vector randomVector(_spaceDim);

	if (_isBounded) {
		for (int i = 0; i < _spaceDim; i++) {
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
int Quantizer::GetSpaceDimension() const {
	return _spaceDim;
}


// Returns the lower bound of the quantizer
Vector Quantizer::GetSpaceLowerBound() const {
	return _spaceLowerBound;
}


// Returns the upper bound of the quantizer
Vector Quantizer::GetSpaceUpperBound() const {
	return _spaceUpperBound;
}