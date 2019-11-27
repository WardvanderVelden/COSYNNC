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
	}
	else {
		_spaceReference = spaceLowerBound;
	}
}


// Quantize a vector to quantization parameters
Vector Quantizer::QuantizeVector(Vector v) {
	Vector quantized(v.GetLength());

	for (int i = 0; i < v.GetLength(); i++) {
		if (_isBounded) {
			quantized[i] = floor(((v[i] - _spaceLowerBound[i]) / _spaceEta[i])) * _spaceEta[i] + _spaceLowerBound[i] + _spaceEta[i] * 0.5;

			if (quantized[i] < _spaceLowerBound[i]) quantized[i] = _spaceLowerBound[i];
			if (quantized[i] > _spaceUpperBound[i]) quantized[i] = _spaceUpperBound[i];
		}
		else {
			quantized[i] = floor(((v[i] - _spaceReference[i]) / _spaceEta[i])) * _spaceEta[i] + _spaceReference[i] + _spaceEta[i] * 0.5;
		}
	}

	return quantized;
}


// Normalize a vector from the bounded space to the normal space
Vector Quantizer::NormalizeVector(Vector v) {
	Vector normal = Vector(v.GetLength());

	for (int i = 0; i < v.GetLength(); i++) {
		// Project back to the bounds
		if (v[i] > (_spaceUpperBound[i] - _spaceEta[i]*0.5)) v[i] = _spaceUpperBound[i] - _spaceEta[i] * 0.5;
		if (v[i] < (_spaceLowerBound[i] + _spaceEta[i]*0.5)) v[i] = _spaceLowerBound[i] + _spaceEta[i] * 0.5;

		// Normalize
		normal[i] = (v[i] - _spaceLowerBound[i]) / (_spaceUpperBound[i] - _spaceLowerBound[i]);
	}

	return normal;
}


// Denormalize a vector a vector from the normal space to the bounded space
Vector Quantizer::DenormalizeVector(Vector v) {
	Vector denormal = Vector(v.GetLength());

	for (int i = 0; i < v.GetLength(); i++) {
		denormal[i] = v[i] * (_spaceUpperBound[i] - _spaceLowerBound[i]) + _spaceLowerBound[i];
	}

	return denormal;
}


// Checks if a vector is in the bounds of the quantized space
Vector Quantizer::IsInBounds(Vector v) {
	for (int i = 0; i < v.GetLength(); i++) {
		if (v[i] < _spaceLowerBound[i] || v[i] > _spaceUpperBound[i]) {
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
			float randomFloat = ((rand() % 100000) / 100000.0)*3.0;
			randomVector[i] = _spaceLowerBound[i] + (_spaceUpperBound[i] - _spaceLowerBound[i]) * randomFloat;
		}
	}

	return QuantizeVector(randomVector);
}