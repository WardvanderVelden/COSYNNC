#pragma once

#include <iostream>
#include <vector>

using namespace std;

namespace COSYNNC {
	class Vector
	{
	public:
		// Initialize a vector by specifying the amount of elements in the vector
		Vector(int length = 0);

		// Initialize a vector by specifying the amount of elements in the vector
		Vector(unsigned int length);

		// Initialize a vector with only a single element
		Vector(double value);

		// Initialize a vector by giving the values for the vector
		Vector(vector<double> values);

		// Mathematical operators
		Vector operator+(const Vector other);
		Vector operator+=(const Vector other);
		Vector operator-(const Vector other);
		Vector operator-=(const Vector other);
		Vector operator*(const double scalar);
		Vector operator*(const Vector other);
		Vector operator/(const double scalar);
		Vector operator/(const Vector other);
		Vector operator=(const vector<double> values);
		bool operator==(const Vector other);

		// Indexing operator
		double& operator[](const int index);

		// Normalizes the vector
		void Normalize();

		// Takes the absolute value of the entries in the vector
		void Abs();

		// Takes the maximum value between itself and the other vector presented
		void Max(Vector other);

		// Returns the dot product of the vector with another vector of the same length
		double Dot(Vector other);

		// Returns the amount of elements in the vector
		int GetLength() const;

		// Sets the amount of elements in the vector
		void SetLength(int length);

		// Returns the norm of the vector
		double GetNorm();

		// Returns the weighted norm of the vector
		double GetWeightedNorm(vector<float> weights);

		// Prints the elements contained within the vector to the console
		void PrintValues() const;

	private:
		int _length = 1;
		vector<double> _values{ 0 };
	};
}