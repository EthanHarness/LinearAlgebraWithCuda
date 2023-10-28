#include "CMatrix.cuh"
#include <vector>
#include <chrono>
#include <stdexcept>

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
	std::vector<std::string> activationFunctions;
	int networkSize;

	NeuralNetwork(int layers[], int size);
	CMatrix processInput(CMatrix inputNodes);
	void stochasticGradDescent(std::vector<CMatrix> trainingData, int epochs, int miniBatchSize, double learningRate, std::vector<CMatrix> testData);
	void updateMiniBatch(CMatrix miniBatch, double learningRate);
	std::vector<CMatrix> backprop(CMatrix networkInput, CMatrix expectedNetworkOutput);
	ActivationFunctionE stringToActivationFunction(const std::string& str);
};
