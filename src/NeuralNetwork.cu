#include "NeuralNetwork.cuh"

//takes in an array of ints and the size of the array
//each value in the array is the size of the corresponding hidden layer
//first value is the size of the input layer. We will not be setting anything for the first layer since its the input. 
NeuralNetwork::NeuralNetwork(int layers[], int size) {
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0, 255.0);

	std::function<double(int, int)> randomNumberGeneratorFunction;
	randomNumberGeneratorFunction = [generator, distribution](int x, int y) mutable {
		return distribution(generator);
	};


	for (int i = 1; i < size; i++) {
		CMatrix weights = createCMatrix(layers[i - 1], layers[i]);
		CMatrix bias = createCMatrix(1, layers[i]);
		std::string activate = "sigmoid";
		setCMatrix(randomNumberGeneratorFunction, weights);
		setCMatrix(randomNumberGeneratorFunction, bias);

		weightsArray.push_back(weights);
		biasArray.push_back(bias);
		activationFunctions.push_back(activate);
	}

	networkSize = size - 1;
}

//Returns the output layer
CMatrix NeuralNetwork::processInput(CMatrix inputNodes) {
	using namespace std;
	CMatrix temp1, temp2, temp3, res;
	temp1 = inputNodes;
	for (int i = 0; i < networkSize; i++, temp1 = res) {

		temp2 = multiply_cuda(temp1, weightsArray[i]);
		temp3 = add_cuda(temp2, biasArray[i]);

		ActivationFunctionE func = stringToActivationFunction(activationFunctions[i]);
		switch (func) {
			case ActivationFunctionE::Sigmoid:
				res = sigmoid_cuda(temp3);
				break;
			case ActivationFunctionE::Tanh:
				res = tanh_cuda(temp3);
				break;
			case ActivationFunctionE::Relu:
				res = relu_cuda(temp3);
				break;
			default:
				throw std::invalid_argument("Unknown activation function found at activationFunctions " + std::to_string(i));
		}

		freeCMatrix(temp1);
		freeCMatrix(temp2);
		freeCMatrix(temp3);
	}

	return res;
}

//Work in progress
void NeuralNetwork::stochasticGradDescent(std::vector<CMatrix> trainingData, int epochs, int miniBatchSize, double learningRate, std::vector<CMatrix> testData) {
	std::random_device rd;
	std::mt19937 g(rd());

	int n = trainingData.size();
	for(int j = 0; j < epochs; j++) {

		//Shuffle training data
		std::shuffle(trainingData.begin(), trainingData.end(), g);
		
		std::vector<std::vector<CMatrix>> miniBatches;
		for(int k = 0; k < n; k += miniBatchSize) {
			int end = std::min(k+miniBatchSize, n);
			std::vector<CMatrix> miniBatch(trainingData.begin() + k, trainingData.end() + end);
			miniBatches.push_back(miniBatch);
		}

	}


}

//Converts a string of our activation function to an enum ActivationFunctionE 
//Probably could refactor this to be more efficient
ActivationFunctionE NeuralNetwork::stringToActivationFunction(const std::string& str) {
    if (str == "sigmoid") return ActivationFunctionE::Sigmoid;
    if (str == "tanh") return ActivationFunctionE::Tanh;
    if (str == "relu") return ActivationFunctionE::Relu;
    return ActivationFunctionE::Unknown;
}