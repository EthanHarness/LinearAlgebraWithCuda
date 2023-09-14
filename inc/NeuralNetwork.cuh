#include "CMatrix.cuh"
#include <vector>
#include <chrono>
#include <stdexcept>

class NeuralNetwork {
public:

	std::vector<CMatrix> weightsArray;
	std::vector<CMatrix> biasArray;
	std::vector<std::string> activationFunctions;
	int networkSize;

	NeuralNetwork(int layers[], int size);
	CMatrix processInput(CMatrix inputNodes);
	double computeLoss(CMatrix computedOutput, CMatrix expectedOutput);
};