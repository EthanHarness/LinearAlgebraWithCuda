#pragma once 
#include "CMatrix.cuh"
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
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
	int evaluate(std::vector<std::pair<CMatrix, CMatrix>> test_data);
	std::pair<std::vector<CMatrix>, std::vector<CMatrix>> backprop(CMatrix networkInput, CMatrix expectedInputsOutput);
	void updateMiniBatch(std::vector<std::pair<CMatrix, CMatrix>> miniBatch, double learningRate);
	void stochasticGradDescent(std::vector<std::pair<CMatrix, CMatrix>> trainingData, int epochs, int miniBatchSize, double learningRate, std::vector<std::pair<CMatrix, CMatrix>> testData);
	
	ActivationFunctionE stringToActivationFunction(const std::string& str);
};
