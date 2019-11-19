#pragma once

#include <iostream>
#include <vector>

using namespace std;

namespace COSYNNC {
	class Vector
	{
	public:
		Vector(int length = 0);
		Vector(vector<float> values);

		Vector operator+(const Vector other);
		Vector operator+=(const Vector other);
		Vector operator-(const Vector other);
		Vector operator-=(const Vector other);
		Vector operator*(const float scalar);
		Vector operator=(const vector<float> values);

		float & operator[](const int index);

		int GetLength() const;
		void SetLength(int length);

		void PrintValues() const;

	private:
		int _length = 1;
		vector<float> _values{ 0 };
	};
}