#include "Quantizer.h"

using namespace COSYNNC;

Quantizer::Quantizer(bool isBounded) {
	_isBounded = isBounded;
}


#pragma region Setters

// Set the quantization parameters for the state space
void Quantizer::SetStateQuantizeParameters(vector<float> stateSpaceEta, vector<float> stateSpaceReference) {
	_stateSpaceDim = stateSpaceEta.size();
	_stateSpaceEta = stateSpaceEta;
	_stateSpaceReference = stateSpaceReference;

	_isBounded = false;
}

void Quantizer::SetStateQuantizeParameters(Vector stateSpaceEta, Vector stateSpaceReference) {
	_stateSpaceDim = stateSpaceEta.GetLength();
	_stateSpaceEta = stateSpaceEta;
	_stateSpaceReference = stateSpaceReference;

	_isBounded = false;
}

// Set the quantization parameters for the state space, if the state space is not bounded the lower bound will be used as the reference
void Quantizer::SetStateQuantizeParameters(Vector stateSpaceEta, Vector stateSpaceLowerBound, Vector stateSpaceUpperBound) {
	_stateSpaceDim = stateSpaceEta.GetLength();
	_stateSpaceEta = stateSpaceEta;
		
	if (_isBounded) {
		_stateSpaceLowerBound = stateSpaceLowerBound;
		_stateSpaceUpperBound = stateSpaceUpperBound;
	}
	else {
		_stateSpaceReference = stateSpaceLowerBound;
	}
}


// Set the quantization parameters for the input space
void Quantizer::SetInputQuantizeParameters(vector<float> inputSpaceEta, vector<float> inputSpaceReference) {
	_inputSpaceDim = inputSpaceEta.size();
	_inputSpaceEta = inputSpaceEta;
	_inputSpaceReference = inputSpaceReference;

	_isBounded = false;
}

void Quantizer::SetInputQuantizeParameters(Vector inputSpaceEta, Vector inputSpaceReference) {
	_inputSpaceDim = inputSpaceEta.GetLength();
	_inputSpaceEta = inputSpaceEta;
	_inputSpaceReference = inputSpaceReference;

	_isBounded = false;
}

// Set the quantization parameters for the input space, if the input space is not bounded the lower bound will be used as the reference
void Quantizer::SetInputQuantizeParameters(Vector inputSpaceEta, Vector inputSpaceLowerBound, Vector inputSpaceUpperBound) {
	_inputSpaceDim = inputSpaceEta.GetLength();
	_inputSpaceEta = inputSpaceEta;

	if (_isBounded) {
		_inputSpaceLowerBound = inputSpaceLowerBound;
		_inputSpaceUpperBound = inputSpaceUpperBound;
	}
	else {
		_inputSpaceReference = inputSpaceLowerBound;
	}
}


#pragma endregion


// Quantize a vector to the quantized state set
Vector Quantizer::QuantizeToState(Vector v) {
	Vector quantized(v.GetLength());

	for (int i = 0; i < v.GetLength(); i++) {
		if (_isBounded) {
			quantized[i] = floor(((v[i] - _stateSpaceLowerBound[i]) / _stateSpaceEta[i])) * _stateSpaceEta[i] + _stateSpaceReference[i] + _stateSpaceEta[i] * 0.5;
		}
		else {
			quantized[i] = floor(((v[i] - _stateSpaceReference[i]) / _stateSpaceEta[i])) * _stateSpaceEta[i] + _stateSpaceReference[i] + _stateSpaceEta[i] * 0.5;
		}
	}

	return quantized;
}


// Quantize a vector to the quantized input set
Vector Quantizer::QuantizeToInput(Vector v) {
	Vector quantized(v.GetLength());

	for (int i = 0; i < v.GetLength(); i++) {
		if (_isBounded) {
			quantized[i] = floor(((v[i] - _inputSpaceLowerBound[i]) / _inputSpaceEta[i])) * _inputSpaceEta[i] + _inputSpaceReference[i] + _inputSpaceEta[i] * 0.5;
		}
		else {
			quantized[i] = floor(((v[i] - _inputSpaceReference[i]) / _inputSpaceEta[i])) * _inputSpaceEta[i] + _inputSpaceReference[i] + _inputSpaceEta[i] * 0.5;
		}
	}

	return quantized;
}