#include "Vector.h"

namespace COSYNNC {
	Vector::Vector(int length) {
		_length = length;
		_values.resize(length, 0);
	}

	Vector::Vector(float value) {
		_length = 1;
		_values.resize(1, value);
	}

	Vector::Vector(vector<float> values) {
		_values = values;
		_length = values.size();
	}


	// Operators
	Vector Vector::operator+(const Vector other) {
		if (_length != other._length) return *this;

		for (int i = 0; i < _length; i++)
			_values[i] += other._values[i];

		return *this;
	}
	Vector Vector::operator+=(const Vector other) { return *this + other; }


	Vector Vector::operator-(const Vector other) {
		if (_length != other._length) return *this;

		for (int i = 0; i < _length; i++)
			_values[i] -= other._values[i];

		return *this;
	}
	Vector Vector::operator-=(const Vector other) { return *this - other; }


	Vector Vector::operator*(const float scalar) {
		for (int i = 0; i < _length; i++)
			_values[i] *= scalar;

		return *this;
	}

	Vector Vector::operator/(const float scalar) {
		return *this * (1 / scalar);
	}


	Vector Vector::operator=(const vector<float> values) {
		_length = values.size();
		_values = values;

		return *this;
	}


	float& Vector::operator[](const int index) {
		return _values[index];
	}

	// Getters and setters
	int Vector::GetLength() const {
		return _length;
	}

	void Vector::SetLength(int size) {
		_values.resize(size, 0);
	}


	void Vector::PrintValues() const {
		for (int i = 0; i < _length; i++) {
			std::cout << _values[i] << " ";
		}
	}
}