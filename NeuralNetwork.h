#pragma once
#include "Random.h"
#include "Globals.h"
#include "ActivationFunctions.h"
#include <unordered_map>
#include <vector>
#include <iostream>
using namespace std;
class NeuralNetwork
{
	class Layer
	{
		private:
			vector<double> neurons; // current layer neuron activations
			vector<vector<double>> weights; // weights connecting the current layer with the next layer
			vector<double> biases; // biases for the neurons of the next layer
			Layer* next_layer;
		public:
			Layer(int size);
			void ConnectWithLayer(Layer* layer);
			vector<double>* GetNeurons();
			void ActivateNext();
			friend class NeuralNetwork;
	};
	private:
		struct Gradient
		{
			vector<double> weight_gradient;
			vector<double> bias_gradient;
		};
		void DoBackpropagation(vector<vector<double>>* training_examples, vector<vector<double>>* expected_outputs);
		Gradient ComputeNegativeGradient(vector<double>* correct_output);
		double FindNeuronDelta(vector<vector<double>>* computed_values, int layer, int neuron_index, vector<double>* correct_output);
		double GetCurrentLoss(vector<double>* expected_output);
		int sgd_sample_size;
		int number_of_weights, number_of_biases;
		double learning_rate;
		vector<Layer> layers;
		vector<vector<double>> *train_set, *test_set;
		vector<vector<double>> *train_set_outputs, *test_set_outputs;
	public:
		NeuralNetwork();
		/// <summary>
		/// Sets number of training examples to use when doing stochastic gradient descent
		/// </summary>
		void SetSampleSize(int size);
		void SetLearningRate(double rate);
		void AddTrainSet(vector<vector<double>>* train_set, vector<vector<double>>* train_set_outputs);
		void AddTestSet(vector<vector<double>>* test_set, vector<vector<double>>* test_set_outputs);
		void Train();
		void AddLayer(int size);
		void SetInput(vector<double> input);
		vector<double> GetOutput();
};

