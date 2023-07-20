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
		CMatrix hiddenLayer = createCMatrix(layers[i], 1);
		CMatrix weights = createCMatrix(layers[i - 1], layers[i]);
		CMatrix bias = createCMatrix(layers[i], 1);
		std::string activate = "sigmoid";
		setCMatrix(randomNumberGeneratorFunction, hiddenLayer);
		setCMatrix(randomNumberGeneratorFunction, weights);
		setCMatrix(randomNumberGeneratorFunction, bias);

		hiddenLayersArray[i] = hiddenLayer;
		weightsArray[i] = weights;
		biasArray[i] = bias;
		activationFunctions[i] = activate;
	}

	networkSize = size - 1;
}