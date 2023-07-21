#include "NeuralNetwork.cuh"

//takes in an array of ints and the size of the array
//each value in the array is the size of the corresponding hidden layer
//first value is the size of the input layer. We will not be setting anything for the first layer since its the input. 
NeuralNetwork::NeuralNetwork(int layers[], int size) {

	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1.0);

	std::function<double(int, int)> randomNumberGeneratorFunction;
	randomNumberGeneratorFunction = [generator, distribution](int x, int y) mutable {
		return distribution(generator);
	};


	for (int i = 1; i < size; i++) {
		CMatrix weights = createCMatrix(layers[i - 1], layers[i]);
		CMatrix bias = createCMatrix(layers[i], 1);
		std::string activate = "sigmoid";
		setCMatrix(randomNumberGeneratorFunction, weights);
		setCMatrix(randomNumberGeneratorFunction, bias);

		weightsArray[i] = weights;
		biasArray[i] = bias;
		activationFunctions[i] = activate;
	}

	networkSize = size - 1;

}

//Returns the output layer
CMatrix NeuralNetwork::processInput(CMatrix inputNodes) {

	CMatrix res = inputNodes;
	for (int i = 0; i < networkSize; i++) {
		res = multiply_cuda(res, weightsArray[i]);
		res = add_cuda(res, biasArray[i]);

		if (activationFunctions[i].compare("sigmoid")) {
			res = sigmoid_cuda(res);
		}
		else if (activationFunctions[i].compare("tanh")) {
			res = tanh_cuda(res);
		}
		else if (activationFunctions[i].compare("relu")) {
			res = relu_cuda(res);
		}
		else {
			throw std::invalid_argument("Unknown activation function found at activationFunctions " + i);
		}
	}

	return res;

}

//double NeuralNetwork::computeLoss(CMatrix computedOutput, CMatrix expectedOutput) {
//
//	CMatrix lossMat = computeLossMatrix_cuda(computedOutput, expectedOutput);
//	double lossValue;
//	for (int i = 0; i < lossMat.width; i++) {
//
//	}
//
//}
