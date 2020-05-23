#include "NeuralNetwork.h"

/// <summary>
/// Adds vector2 to vector1, modifies vector1
/// </summary>
void AddVectors(vector<double>* vector1, vector<double>* vector2)
{
	for (int i = 0; i < vector1->size(); i++)
	{
		vector1->at(i) += vector2->at(i);
	}
}

void NeuralNetwork::DoBackpropagation(vector<vector<double>>* training_examples, vector<vector<double>>* expected_outputs)
{
	Gradient average_negative_gradient = { vector<double>(number_of_weights), vector<double>(number_of_biases) };
	double loss_sum = 0;

	//Compute average 
	for (int example_index = 0; example_index < training_examples->size(); example_index++)
	{
		SetInput(training_examples->at(example_index));
		GetOutput();
		loss_sum += GetCurrentLoss(&expected_outputs->at(example_index));

		Gradient stochastic_gradient = ComputeNegativeGradient(&expected_outputs->at(example_index));

		AddVectors(&average_negative_gradient.weight_gradient, &stochastic_gradient.weight_gradient);
		AddVectors(&average_negative_gradient.bias_gradient, &stochastic_gradient.bias_gradient);
	}

	cout << "Average loss: " << loss_sum / training_examples->size() << "\n";

	for (int i = 0; i < average_negative_gradient.bias_gradient.size(); i++)
		average_negative_gradient.bias_gradient[i] /= training_examples->size();

	//Update weights
	int gradient_index = 0;
	for (int layer_index = 0; layer_index < layers.size(); layer_index++)
	{
		for (int neuron_index = 0; neuron_index < layers[layer_index].weights.size(); neuron_index++)
		{
			for (int weight_index = 0; weight_index < layers[layer_index].weights[neuron_index].size(); weight_index++)
			{
				layers[layer_index].weights[neuron_index][weight_index] += average_negative_gradient.weight_gradient[gradient_index++];
			}
		}
	}

	//Update biases
	gradient_index = 0;
	for (int layer_index = 0; layer_index < layers.size(); layer_index++)
	{
		for (int bias_index = 0; bias_index < layers[layer_index].biases.size(); bias_index++)
		{
			layers[layer_index].biases[bias_index] += average_negative_gradient.bias_gradient[gradient_index++];
		}
	}
}
NeuralNetwork::Gradient NeuralNetwork::ComputeNegativeGradient(vector<double>* correct_output)
{
	Gradient negative_gradient;

	negative_gradient.weight_gradient.reserve(number_of_weights);
	negative_gradient.bias_gradient.reserve(number_of_biases);

	//Initialize all values to INFINITY
	vector<vector<double>> computed_values = vector<vector<double>>(layers.size());
	for (int i = 0; i < computed_values.size(); i++)
		computed_values[i] = vector<double>(layers[i].neurons.size(),INFINITY);

	//Compute negative weight gradient
	for (int layer_index = 1; layer_index < layers.size(); layer_index++)
	{
		for (int neuron_index = 0; neuron_index < layers[layer_index-1].neurons.size(); neuron_index++)
		{
			double left_neuron_activation = layers[layer_index-1].neurons[neuron_index];
			for (int right_neuron_index = 0; right_neuron_index < layers[layer_index-1].weights[neuron_index].size(); right_neuron_index++)
			{
				double neuron_delta = FindNeuronDelta(&computed_values, layer_index, right_neuron_index, correct_output);
				negative_gradient.weight_gradient.push_back((-1) * left_neuron_activation * neuron_delta * learning_rate);
			}
		}
		for (int right_neuron_index = 0; right_neuron_index < layers[layer_index-1].biases.size(); right_neuron_index++)
		{
			double neuron_delta = FindNeuronDelta(&computed_values, layer_index, right_neuron_index, correct_output);
			negative_gradient.bias_gradient.push_back((-1) * neuron_delta * learning_rate);
		}
	}

	return negative_gradient;
}
double NeuralNetwork::FindNeuronDelta(vector<vector<double>>* computed_values, int layer, int neuron_index, vector<double>* correct_output)
{
	if (computed_values->at(layer)[neuron_index] != INFINITY)
	{
		return computed_values->at(layer)[neuron_index];
	}

	double left_neuron_activation = layers[layer-1].neurons[neuron_index];
	double right_neuron_activation = layers[layer].neurons[neuron_index];
	if (layer == layers.size()-1) //Neuron is in output layer
	{
		//Derivative of loss function
		double delta = (right_neuron_activation - correct_output->at(neuron_index));
		//Derivative of activation function
		delta *= right_neuron_activation * (1 - right_neuron_activation);
		return delta;
	}
	else
	{
		double sum = 0;
		for (int next_neuron_index = 0; next_neuron_index < layers[layer+1].neurons.size(); next_neuron_index++)
		{
			double connecting_weight = layers[layer].weights[neuron_index][next_neuron_index];
			double next_neuron_delta = FindNeuronDelta(computed_values, layer + 1, next_neuron_index, correct_output);
			sum += connecting_weight * next_neuron_delta;
		}
		
		double delta = sum * right_neuron_activation * (1 - right_neuron_activation);
		computed_values->at(layer)[neuron_index] = delta;
		return delta;
	}
}

double NeuralNetwork::GetCurrentLoss(vector<double>* expected_output)
{
	double loss_sum = 0;
	int output_neurons_count = layers.back().neurons.size();
	for (int neuron_index = 0; neuron_index < output_neurons_count; neuron_index++)
	{
		double difference = expected_output->at(neuron_index) - layers.back().neurons[neuron_index];
		loss_sum += difference * difference * 0.5;
	}
	return loss_sum;
}

NeuralNetwork::NeuralNetwork()
{
	sgd_sample_size = 100;
	learning_rate = 1;
}

void NeuralNetwork::SetSampleSize(int size)
{
	sgd_sample_size = size;
}

void NeuralNetwork::SetLearningRate(double rate)
{
	learning_rate = rate;
}

void NeuralNetwork::AddTrainSet(vector<vector<double>>* train_set, vector<vector<double>>* train_set_outputs)
{
	this->train_set_outputs = train_set_outputs;
	this->train_set = train_set;
}

void NeuralNetwork::AddTestSet(vector<vector<double>>* test_set, vector<vector<double>>* test_set_outputs)
{
	this->test_set_outputs = test_set_outputs;
	this->test_set = test_set;
}

void NeuralNetwork::Train()
{
	for (int i = 0; i < layers.size() - 1; i++)
		layers[i].ConnectWithLayer(&layers[i + 1]);

	number_of_weights = 0;
	for (int i = 0; i < layers.size(); i++)
	{
		for (int j = 0; j < layers[i].weights.size(); j++)
		{
			number_of_weights += layers[i].weights[j].size();
		}
	}
	number_of_biases = 0;
	for (int i = 0; i < layers.size(); i++)
		number_of_biases += layers[i].biases.size();
	
	int no_test_examples = (test_set != nullptr) ? test_set->size() : 0;
	int no_train_examples = train_set->size();

	//Do stochastic gradient descent
	for (int i = 0; i < no_train_examples; i+=sgd_sample_size)
	{
		auto training_examples = vector<vector<double>>(train_set->begin() + i, train_set->begin() + i + sgd_sample_size);
		auto expected_outputs = vector<vector<double>>(train_set_outputs->begin() + i, train_set_outputs->begin() + i + sgd_sample_size);
		DoBackpropagation(&training_examples, &expected_outputs);
	}

	int wrong_count = 0;
	for (int i = 0; i < no_test_examples; i++)
	{
		vector<double> input = test_set->at(i);
		vector<double> output = test_set_outputs->at(i);

		SetInput(input);
		vector<double> output_layer = GetOutput();
		double loss = GetCurrentLoss(&output);

		int prediction = -1;
		double max_output_activation = -INFINITY;
		for (int i = 0; i < output_layer.size(); i++)
		{
			if (output_layer[i] > max_output_activation)
			{
				max_output_activation = output_layer[i];
				prediction = i;
			}
		}

		int correct_label = -1;
		for (int i = 0; i < output.size(); i++)
		{
			if (output[i] > 0.99)
			{
				correct_label = i;
			}
		}

		if (correct_label != prediction)
		{
			wrong_count++;

			/*cout << "Correct output: ";
			for (int i = 0; i < output.size(); i++)
			{
				cout << i << ": " << output[i] << " ";
			}
			cout << "\n";

			cout << "NN output: ";
			for (int i = 0; i < output.size(); i++)
			{
				cout << i << ": " << output_layer[i] << " ";
			}
			cout << "\n";
			cout << "Loss: " << loss << "\n";
			cout << "Prediction: " << prediction << "\n" << " Correct label: " << correct_label << "\n";*/

		}
	}
	cout << "Error rate: " << wrong_count/(1.0*test_set->size())*100 << "%\n";
	//cout << "Number of correctly classified test examples: " << test_set->size() - wrong_count << " out of " << test_set->size() << "\n";
}

vector<double> NeuralNetwork::GetOutput()
{	
	for (int i = 0; i < layers.size()-1; i++)
		layers[i].ActivateNext();
	
	return layers.back().neurons;
}

void NeuralNetwork::AddLayer(int size)
{
	Layer l = Layer(size);
	layers.push_back(l);
}

void NeuralNetwork::SetInput(vector<double> input)
{
	layers[0].neurons = input;
}

NeuralNetwork::Layer::Layer(int size)
{
	neurons = vector<double>(size);
}

void NeuralNetwork::Layer::ConnectWithLayer(Layer* next_layer)
{
	this->next_layer = next_layer;

	//Initialize weights
	int next_layer_size = next_layer->GetNeurons()->size();
	weights = vector<vector<double>>(neurons.size());
	for (int i = 0; i < weights.size(); i++)
	{
		weights[i] = vector<double>(next_layer_size);
		for (int j = 0; j < next_layer_size; j++)
			weights[i][j] = Globals::random_generator.GetHeSigmoidWeight(neurons.size(), next_layer_size);
	}

	//Initialize biases
	biases = vector<double>(next_layer_size);
}

void NeuralNetwork::Layer::ActivateNext()
{

	auto next_layer_neurons = next_layer->GetNeurons();

	//Add product of weights and activations of current layer to next layer
	for (int neuron_index = 0; neuron_index < neurons.size(); neuron_index++)
	{
		double current_activation = neurons[neuron_index];

		//For each neuron add product of activation and weight to corresponding neuron in next layer
		for (int weight_index = 0; weight_index < next_layer_neurons->size(); weight_index++)
		{
			next_layer_neurons->at(weight_index) += current_activation * weights[neuron_index][weight_index];
		}
	}

	//Add biases to next layer activation
	for (int bias_index = 0; bias_index < biases.size(); bias_index++)
		next_layer_neurons->at(bias_index) += biases[bias_index];
	
	//Apply activation function to the next layer
	for (int neuron_index = 0; neuron_index < next_layer_neurons->size(); neuron_index++)
		next_layer_neurons->at(neuron_index) = ActivationFunctions::Sigmoid(next_layer_neurons->at(neuron_index));
}

vector<double>* NeuralNetwork::Layer::GetNeurons()
{
	return &neurons;
}
