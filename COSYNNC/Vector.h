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

		// Initialize a vector with only a single element
		Vector(float value);

		// Initialize a vector by giving the values for the vector
		Vector(vector<float> values);

		// Mathematical operators
		Vector operator+(const Vector other);
		Vector operator+=(const Vector other);
		Vector operator-(const Vector other);
		Vector operator-=(const Vector other);
		Vector operator*(const float scalar);
		Vector operator/(const float scalar);
		Vector operator=(const vector<float> values);
		bool operator==(const Vector other);

		// Indexing operator
		float& operator[](const int index);

		// Returns the amount of elements in the vector
		int GetLength() const;

		// Sets the amount of elements in the vector
		void SetLength(int length);

		// Returns the norm of the vector
		float GetNorm();

		// Returns the weighted norm of the vector
		float GetWeightedNorm(vector<float> weights);

		// Prints the elements contained within the vector to the console
		void PrintValues() const;

	private:
		int _length = 1;
		vector<float> _values{ 0 };
	};
}