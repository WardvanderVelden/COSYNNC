#pragma once

#include <vector>
#include <string>

using namespace std;

namespace COSYNNC {
	class StringHelper {
	public:
		// Replace the denominated characters with the replacement onces in a string
		virtual void ReplaceAll(string &str, char denominator, char replacement = 0x00);

		// Split the string into a vector of strings based on a denominator
		virtual vector<string> Split(string str, char denominator);
	};
};