#pragma once
#include <math.h>
#include "Vector.h";
#include "Quantizer.h";
#include "Edge.h"

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
		Quantizer();

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

		// Checks if a vector is in the bounds of the quantized space
		bool IsInBounds(Vector vector);

		#pragma region Getters

		// Returns the vector that corresponds to the labelled output of the network
		Vector GetVectorFromOneHot(Vector oneHot);

		// Returns the vector that corresponds to the index in the space
		Vector GetVectorFromIndex(long index);

		// Returns the axis indices 
		vector<unsigned long> GetAxisIndicesFromIndex(unsigned long index);

		// Returns the index that corresponds to that vector
		long GetIndexFromVector(Vector vector);

		// Gets a random vector within the bounded quantization space if isBounded is set
		Vector GetRandomVector();

		// Returns the cardinality of the quantized set
		long GetCardinality() const;

		// Returns the dimension of the space
		int GetDimension() const;

		// Returns the lower bound of the quantizer
		Vector GetLowerBound() const;

		// Returns the upper bound of the quantizer
		Vector GetUpperBound() const;

		// Returns the space eta of the quantizer
		Vector GetEta() const;

		// Returns an array of vectors which are the vertices of the hyper cell
		Vector* GetCellVertices(Vector cell);
		Vector* GetCellVertices(unsigned long cellIndex);

		#pragma endregion Getters		
	private:
		int _spaceDimension = 0;

		vector<long> _spaceCardinalityPerAxis;
		long _spaceCardinality = 0;

		Vector _spaceEta;

		Vector _spaceLowerVertex;
		Vector _spaceUpperVertex;
	};
}

