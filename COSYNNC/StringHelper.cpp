#include "StringHelper.h"

namespace COSYNNC {
	// Replace the denominated characters with the replacement onces in a string
	void StringHelper::ReplaceAll(string& str, char denominator, char replacement) {
		auto pos = str.find(denominator);
		while (pos != -1) {
			if (replacement == NULL) str.erase(pos, 1);
			else str[pos] = replacement;
			pos = str.find(denominator);
		}
	}

	// Split the string into a vector of strings based on a denominator
	vector<string> StringHelper::Split(string str, char denominator) {
		vector<string> vec;

		auto pos = str.find(denominator);
		while (pos != -1) {
			vec.push_back(str.substr(0, pos));
			str = str.substr(pos + 1, str.size() - pos - 1);

			pos = str.find(denominator);
		}

		if (str.size() >= 1) {
			vec.push_back(str.substr(0, pos));
		}

		return vec;
	}
}