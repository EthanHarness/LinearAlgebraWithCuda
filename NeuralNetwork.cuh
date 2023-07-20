#include "CMatrix.cuh"
#include <vector>
#include <chrono>

class NeuralNetwork {
public:

	std::vector<CMatrix> weightsArray;
	std::vector<CMatrix> hiddenLayersArray;
	std::vector<CMatrix> biasArray;
	std::vector<std::string> activationFunctions;
	int networkSize;

	NeuralNetwork(int layers[], int size);
};