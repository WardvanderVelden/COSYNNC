#pragma once
#include <math.h>
#include "Vector.h";

namespace COSYNNC {
	class Quantizer
	{
	public:
		// Initialized the quantizer, by default isBounded is set to false so that only a reference point needs to be specified
		Quantizer(bool isBounded = false);


		// Set the quantization parameters for the space
		void SetQuantizeParameters(Vector stateSpaceEta, Vector stateSpaceReference);

		// Set the quantization parameters for the space, if the space is not bounded the lower bound will be used as the reference
		void SetQuantizeParameters(Vector stateSpaceEta, Vector stateSpaceLowerBound, Vector stateSpaceUpperBound);

		// Set quantize a vector to the quantized space based on the quantization parameters
		Vector QuantizeVector(Vector v);

		// Normalize a vector from the bounded space to the normal space
		Vector NormalizeVector(Vector v);

		// Denormalize a vector a vector from the normal space to the bounded space
		Vector DenormalizeVector(Vector v);

		// Checks if a vector is in the bounds of the quantized space
		Vector IsInBounds(Vector v);

		// Gets a random vector within the bounded quantization space if isBounded is set
		Vector GetRandomVector();

	private:
		bool _isBounded = false;

		int _spaceDim = 0;
		int _inputSpaceDim = 0;

		Vector _spaceEta;

		Vector _spaceLowerBound;
		Vector _spaceUpperBound;

		Vector _spaceReference;
	};
}

