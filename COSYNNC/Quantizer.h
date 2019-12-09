#pragma once
#include <math.h>
#include "Vector.h";
#include "Quantizer.h";

# define PI 3.14159265358979323846

namespace COSYNNC {
	struct ProbabilisticVector {
		Vector vector;
		float probability;
	};

	class Quantizer
	{
	public:
		// Initialized the quantizer, by default isBounded is set to false so that only a reference point needs to be specified
		Quantizer(bool isBounded = false);


		// Set the quantization parameters for the space
		void SetQuantizeParameters(Vector spaceEta, Vector spaceReference);

		// Set the quantization parameters for the space, if the space is not bounded the lower bound will be used as the reference
		void SetQuantizeParameters(Vector spaceEta, Vector spaceLowerBound, Vector spaceUpperBound);

		// Quantize a vector to the quantized space based on the quantization parameters
		Vector QuantizeVector(Vector vector);

		// Quantizes a normalized vector to a normalized quantized vector
		Vector QuantizeNormalizedVector(Vector vector);

		// Normalize a vector from the bounded space to the normal space
		Vector NormalizeVector(Vector denormal);

		// Denormalize a vector a vector from the normal space to the bounded space
		Vector DenormalizeVector(Vector normal);

		// Returns the nearest quantized element in the quantized space and returns its probability, also provides a random alternative
		vector<ProbabilisticVector> QuantizeVectorProbabilistically(Vector denormal);

		// Returns the input that corresponds to the labelled output of the network
		Vector FindVectorFromOneHot(Vector oneHot);

		// Checks if a vector is in the bounds of the quantized space
		bool IsInBounds(Vector vector);

		// Gets a random vector within the bounded quantization space if isBounded is set
		Vector GetRandomVector();

		// Returns the cardinality of the quantized set
		int GetCardinality() const;

	private:
		bool _isBounded = false;

		int _spaceDim = 0;
		int _inputSpaceDim = 0;

		vector<int> _spaceCardinalityPerAxis;
		int _spaceCardinality = 0;

		Vector _spaceEta;
		double _spaceEtaEllipsoidVolume = 0.0;

		Vector _spaceLowerBound;
		Vector _spaceUpperBound;

		Vector _spaceReference;
	};
}

