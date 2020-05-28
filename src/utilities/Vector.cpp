#include "Vector.h"

namespace COSYNNC {
	Vector::Vector(int length) {
		_length = length;
		_values.resize(length, 0);
	}

	Vector::Vector(unsigned int length) {
		_length = length;
		_values.resize(length, 0);
	}

	Vector::Vector(double value) {
		_length = 1;
		_values.resize(1);
		_values[0] = value;
	}

	Vector::Vector(vector<double> values) {
		_values = values;
		_length = values.size();
	}


	// Operators
	Vector Vector::operator+(const Vector other) {
		if (_length != other._length) return *this;

		Vector vec(*this);

		for (int i = 0; i < _length; i++)
			vec[i] += other._values[i];

		return vec;
	}
	Vector Vector::operator+=(const Vector other) { return *this + other; }

	Vector Vector::operator-(const Vector other) {
		if (_length != other._length) return *this;

		Vector vec(*this);

		for (int i = 0; i < _length; i++)
			vec[i] -= other._values[i];

		return vec;
	}
	Vector Vector::operator-=(const Vector other) { return *this - other; }

	Vector Vector::operator*(const double scalar) {
		Vector vec(*this);

		for (int i = 0; i < _length; i++)
			vec[i] *= scalar;

		return vec;
	}
	Vector Vector::operator*(const Vector other) {
		Vector vec(*this);

		for (int i = 0; i < _length; i++)
			vec[i] *= other._values[i];

		return vec;
	}

	Vector Vector::operator/(const double scalar) {
		return *this * (1 / scalar);
	}
	Vector Vector::operator/(const Vector other) {
		Vector vec(*this);

		for (int i = 0; i < _length; i++)
			vec[i] /= other._values[i];

		return vec;
	}

	Vector Vector::operator=(const vector<double> values) {
		_length = values.size();
		_values = values;

		return *this;
	}
	bool Vector::operator==(const Vector other) {
		if (_length != other._length) return false;

		for (int i = 0; i < _length; i++) {
			if (_values[i] != other._values[i])	return false;
		}

		return true;
	}

	double& Vector::operator[](const int index) {
		return _values[index];
	}


	// Normalizes the vector
	void Vector::Normalize() {
		auto norm = GetNorm();
		for (unsigned int i = 0; i < _length; i++) {
			_values[i] = _values[i] / norm;
		}
	}


	// Takes the absolute value of the entries in the vector
	void Vector::Abs() {
		for (unsigned int i = 0; i < _length; i++) {
			_values[i] = abs(_values[i]);
		}
	}


	// Takes the maximum value between itself and the other vector presented
	void Vector::Max(Vector other) {
		if (_length != other.GetLength()) return;
		for (unsigned int i = 0; i < _length; i++) {
			if (_values[i] < other[i]) _values[i] = other[i];
		}
	}


	// Returns the dot product of the vector with another vector of the same length
	double Vector::Dot(Vector other) {
		if (_length != other.GetLength()) return 0.0;

		double value = 0.0;
		for (unsigned int i = 0; i < _length; i++) {
			value += (_values[i] * other[i]);
		}

		return value;
	}


	// Getters and setters
	int Vector::GetLength() const {
		return _length;
	}


	void Vector::SetLength(int size) {
		_values.resize(size, 0);
	}


	// Returns the norm of the vector
	double Vector::GetNorm() {
		double norm = 0.0;
		for (int i = 0; i < _length; i++) {
			norm += pow(_values[i], 2);
		}
		return sqrt(norm);
	}


	// Returns the weighted norm of the vector
	double Vector::GetWeightedNorm(vector<float> weights) {
		double norm = 0.0;
		for (int i = 0; i < _length; i++) {
			norm += pow(weights[i] * _values[i], 2);
		}
		return sqrt(norm);
	}


	void Vector::PrintValues() const {
		for (int i = 0; i < _length; i++) {
			std::cout << _values[i] << " ";
		}
	}
}