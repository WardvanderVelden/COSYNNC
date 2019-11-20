#include "Quantizer.h"

using namespace COSYNNC;

Quantizer::Quantizer(bool isBounded) {
	_isBounded = isBounded;
}


// Set the quantization parameters for the space
void Quantizer::SetQuantizeParameters(vector<float> spaceEta, vector<float> spaceReference) {
	_spaceDim = spaceEta.size();
	_spaceEta = Vector(spaceEta);
	_spaceReference = Vector(spaceReference);

	_isBounded = false;
}

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
			quantized[i] = floor(((v[i] - _spaceLowerBound[i]) / _spaceEta[i])) * _spaceEta[i] + _spaceReference[i] + _spaceEta[i] * 0.5;
		}
		else {
			quantized[i] = floor(((v[i] - _spaceReference[i]) / _spaceEta[i])) * _spaceEta[i] + _spaceReference[i] + _spaceEta[i] * 0.5;
		}
	}

	return quantized;
}