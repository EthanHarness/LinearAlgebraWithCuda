#pragma once 
#include "CMatrix.cuh"
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <ExceptionWithDetails.h>

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
	int evaluate(std::vector<std::pair<CMatrix, int>> test_data);
	std::pair<std::vector<CMatrix>, std::vector<CMatrix>> backprop(CMatrix networkInput, int expectedInputsOutput);
	void updateMiniBatch(std::vector<std::pair<CMatrix, int>> miniBatch, double learningRate);
	void stochasticGradDescent(std::vector<std::pair<CMatrix, int>> trainingData, int epochs, int miniBatchSize, double learningRate, std::vector<std::pair<CMatrix, int>> testData);
	
	ActivationFunctionE stringToActivationFunction(const std::string& str);
};
