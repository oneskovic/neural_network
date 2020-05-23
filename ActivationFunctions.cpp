#include "ActivationFunctions.h"

double ActivationFunctions::Sigmoid(double activation)
{
    return 1.0 / (1+exp(-activation));
}

double ActivationFunctions::Relu(double activation)
{
	if (activation < 0)
		return 0;
	else
		return activation;
}
