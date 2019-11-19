#pragma once
#include "Vector.h";
#include "Plant.h"
#include "Quantizer.h"

namespace COSYNNC {
	class Controller
	{
	public:
		Controller();
		Controller(Plant* plant);
		Controller(Plant* plant, Quantizer* quantizer);

		Vector GetControlAction(Vector state);
	private:
		Vector _lastControlAction;

		int _stateSpaceDim;
		int _inputSpaceDim;

		Quantizer* _quantizer = NULL;
	};
}


