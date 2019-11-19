#pragma once
#include <math.h>
#include "Vector.h";

namespace COSYNNC {
	class Quantizer
	{
	public:
		Quantizer(bool isBounded = false);

		void SetStateQuantizeParameters(vector<float> stateSpaceEta, vector<float> stateSpaceReference);
		void SetStateQuantizeParameters(Vector stateSpaceEta, Vector stateSpaceReference);
		void SetStateQuantizeParameters(Vector stateSpaceEta, Vector stateSpaceLowerBound, Vector stateSpaceUpperBound);

		void SetInputQuantizeParameters(vector<float> inputSpaceEta, vector<float> inputSpaceReference);
		void SetInputQuantizeParameters(Vector inputSpaceEta, Vector inputSpaceReference);
		void SetInputQuantizeParameters(Vector inputSpaceEta, Vector inputSpaceLowerBound, Vector inputSpaceUpperBound);

		Vector QuantizeToState(Vector v);
		Vector QuantizeToInput(Vector v);

	private:
		bool _isBounded = false;

		int _stateSpaceDim = 0;
		int _inputSpaceDim = 0;

		Vector _stateSpaceEta;
		Vector _inputSpaceEta;

		Vector _stateSpaceLowerBound;
		Vector _stateSpaceUpperBound;

		Vector _inputSpaceLowerBound;
		Vector _inputSpaceUpperBound;

		Vector _stateSpaceReference;
		Vector _inputSpaceReference;
	};
}

