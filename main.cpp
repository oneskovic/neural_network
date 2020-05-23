#include <iostream>
#include <string>
#include <fstream>
#include "NeuralNetwork.h"
using namespace std;

int main()
{
	Globals::random_generator = Random();

	vector<vector<double>> training_set, training_set_outputs;
	vector<vector<double>> test_set, test_set_outputs;

	string input_path = "C:/Users/Fizz2/Desktop/Programiranje/MasinskoUcenje/MNIST/train_set2.txt";
	ifstream input_stream(input_path);
	string input;
	for(int j = 0; j < 50000; j++)
	{
		vector<double> input_vec = vector<double>(28 * 28);
		for (int i = 0; i < 28 * 28; i++)
		{
			input_stream >> input;
			input_vec[i] = stoi(input)/255.0;
		}
		training_set.push_back(input_vec);

		string label;
		input_stream >> label;
		int label_int = stoi(label);
		vector<double> output(10);
		output[label_int] = 1;
		training_set_outputs.push_back(output);
	}

	for (int j = 0; j < 10000; j++)
	{
		vector<double> input_vec = vector<double>(28 * 28);
		for (int i = 0; i < 28 * 28; i++)
		{
			input_stream >> input;
			input_vec[i] = stoi(input)/255.0;
		}
		test_set.push_back(input_vec);

		string label;
		input_stream >> label;
		int label_int = stoi(label);
		vector<double> output(10);
		output[label_int] = 1;
		test_set_outputs.push_back(output);
	}
	while (true)
	{
		double learning_rate;
		int sample_size;
		cin >> learning_rate;
		if (learning_rate < 0)
		{
			return 0;
		}
		cin >> sample_size;
		NeuralNetwork nn = NeuralNetwork();
		nn.AddLayer(28 * 28);
		nn.AddLayer(64);
		nn.AddLayer(64);
		nn.AddLayer(10);
		nn.AddTrainSet(&training_set, &training_set_outputs);
		nn.AddTestSet(&test_set, &test_set_outputs);
		nn.SetLearningRate(learning_rate);
		nn.SetSampleSize(sample_size);
		nn.Train();
	}	
	return 0;
}