#pragma once 
#include "CMatrix.cuh"
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>

enum class ActivationFunctionE {
    Sigmoid,
    Tanh,
    Relu,
    Unknown
};


class NeuralNetwork {
public:
	std::vector<CMatrix> weightsArray;
	std::vector<CMatrix> biasArray;
	std::vector<ActivationFunctionE> activationFunctions;
	int networkSize;

	NeuralNetwork(int layers[], int size);
	CMatrix processInput(CMatrix inputNodes);
	void stochasticGradDescent(std::vector<std::pair<CMatrix, int>> trainingData, int epochs, int miniBatchSize, double learningRate, std::vector<std::pair<CMatrix, int>> testData);
	void updateMiniBatch(std::vector<std::pair<CMatrix, int>>, double learningRate);
	std::vector<CMatrix> backprop(CMatrix networkInput, int expectedInputsOutput);
	int evaluate(std::vector<std::pair<CMatrix, int>>);
	
	ActivationFunctionE stringToActivationFunction(const std::string& str);
};
