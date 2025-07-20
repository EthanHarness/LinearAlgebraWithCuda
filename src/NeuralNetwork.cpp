#include "NeuralNetwork.h"

const std::function<double(int, int)> zeroFunc = [](int x, int y) {
	return 0;
	};

//takes in an array of ints and the size of the array
//each value in the array is the size of the corresponding hidden layer
//first value is the size of the input layer. We will not be setting anything for the first layer since its the input. 
NeuralNetwork::NeuralNetwork(int layers[], int size) {
	const double variance = 1;
	const double mean = 0;
	const ActivationFunctionE temporaryActivationFunctionConstant = ActivationFunctionE::Sigmoid;
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(mean, variance);

	std::function<double(int, int)> randomNumberGeneratorFunction;
	randomNumberGeneratorFunction = [generator, distribution](int x, int y) mutable {
		return distribution(generator);
		};


	for (int i = 1; i < size; i++) {
		CMatrix weights = createCMatrix(layers[i], layers[i - 1]);
		CMatrix bias = createCMatrix(layers[i], 1);
		setCMatrix(randomNumberGeneratorFunction, weights);
		setCMatrix(randomNumberGeneratorFunction, bias);

		weightsArray.push_back(weights);
		biasArray.push_back(bias);
		activationFunctions.push_back(temporaryActivationFunctionConstant);
	}

	networkSize = size - 1;
}

//Returns the output layer based off the networks current weights and biases.
//DOES NOT MODIFY OR ADJUST THOSE WEIGHTS AND BIASES.
CMatrix NeuralNetwork::processInput(CMatrix inputNodes) {
	using namespace std;
	CMatrix res = inputNodes;
	for (int i = 0; i < networkSize; i++) {
		res = add_cuda(multiply_cuda(weightsArray[i], res), biasArray[i]);

		switch (activationFunctions[i]) {
		case ActivationFunctionE::Sigmoid:
			res = sigmoid_cuda(res);
			break;
		case ActivationFunctionE::Tanh:
			res = tanh_cuda(res);
			break;
		case ActivationFunctionE::Relu:
			res = relu_cuda(res);
			break;
		case ActivationFunctionE::Unknown:
			throw std::invalid_argument("Unknown activation function found at activationFunctions " + std::to_string(i));
		default:
			throw std::invalid_argument("Unknown activation function found at activationFunctions " + std::to_string(i));
		}
	}

	return res;
}

//Evaluates the accuracy of the network
int NeuralNetwork::evaluate(std::vector<std::pair<CMatrix, int>> test_data) {

	std::vector<std::pair<int, int>> results;
	int sum = 0;
	for (auto twoTuple : test_data) {
		CMatrix inputNodes = twoTuple.first;
		int expectedOutput = twoTuple.second;
		results.push_back(std::pair<int, int>{getArgmax(processInput(inputNodes)), expectedOutput});
	}
	for (auto twoTuple : results) {
		int computed = twoTuple.first;
		int realResult = twoTuple.second;
		if (computed == realResult) sum++;
	}
	return sum;
}

//Implements back prop algo
std::pair<std::vector<CMatrix>, std::vector<CMatrix>> NeuralNetwork::backprop(CMatrix networkInput, int expectedInputsOutput) {
	std::vector<CMatrix> nabla_b;
	std::vector<CMatrix> nabla_w;
	for (auto bias : biasArray) {
		CMatrix t = createCMatrix(bias.height, bias.width);
		setCMatrix(zeroFunc, t);
		nabla_b.push_back(t);
	}
	for (auto weights : weightsArray) {
		CMatrix t = createCMatrix(weights.height, weights.width);
		setCMatrix(zeroFunc, t);
		nabla_w.push_back(t);
	}

	CMatrix currentLayerActivation = networkInput;
	std::vector<CMatrix> computedLayerFinalResult{ networkInput }; //Also contains first layer
	std::vector<CMatrix> computedLayerIntermediateResult;
	for (int i = 0; i < biasArray.size(); i++) {
		if (biasArray[i].height != weightsArray[i].height) {
			std::ostringstream oss;
			oss << "Somehow someway the bias array in layer " << i << " does not match the weights array";
			std::string errorMsg = oss.str();
			throw_line(errorMsg);
		}

		CMatrix intermediate = add_cuda(multiply_cuda(weightsArray[i], currentLayerActivation), biasArray[i]);
		computedLayerIntermediateResult.push_back(intermediate);
		currentLayerActivation = sigmoid_cuda(intermediate);
		computedLayerFinalResult.push_back(currentLayerActivation);
	}

	CMatrix delta = emultiply_cuda(sadd_cuda(computedLayerFinalResult.back(), (expectedInputsOutput * -1)), sigmoid_prime_cuda(computedLayerIntermediateResult.back()));
	nabla_b.back() = delta;
	nabla_w.back() = multiply_cuda(delta, transpose_cuda(computedLayerFinalResult[computedLayerFinalResult.size() - 2]));

	for (int j = 2; j < networkSize + 1;j++) {
		CMatrix intermediate = computedLayerIntermediateResult[computedLayerIntermediateResult.size() - j];
		CMatrix sigPrime = sigmoid_prime_cuda(intermediate);
		delta = emultiply_cuda(multiply_cuda(transpose_cuda(weightsArray[weightsArray.size() - j + 1]), delta), sigPrime);
		nabla_b[nabla_b.size() - j] = delta;
		nabla_w[nabla_w.size() - j] = multiply_cuda(delta, transpose_cuda(computedLayerFinalResult[computedLayerFinalResult.size() - j - 1]));
	}
	return std::pair<std::vector<CMatrix>, std::vector<CMatrix>>{nabla_b, nabla_w};
}

//This function updates the weights and biasies according to the output of the backprop algo
void NeuralNetwork::updateMiniBatch(std::vector<std::pair<CMatrix, int>> miniBatch, double learningRate) {

	std::vector<CMatrix> nabla_b;
	for (auto bias : biasArray) {
		CMatrix t = createCMatrix(bias.height, bias.width);
		setCMatrix(zeroFunc, t);
		nabla_b.push_back(t);
	}
	std::vector<CMatrix> nabla_w;
	for (auto weights : weightsArray) {
		CMatrix t = createCMatrix(weights.height, weights.width);
		setCMatrix(zeroFunc, t);
		nabla_w.push_back(t);
	}

	for (auto io : miniBatch) {
		CMatrix input = io.first;
		int expectedOutput = io.second;
		auto output = backprop(input, expectedOutput);
		std::vector<CMatrix> delta_nabla_b = output.first;
		std::vector<CMatrix> delta_nabla_w = output.second;

		if (nabla_b.front().height != delta_nabla_b.front().height || nabla_b.front().width != delta_nabla_b.front().width ||
			nabla_w.front().height != delta_nabla_w.front().height || nabla_w.front().width != delta_nabla_w.front().width)
			throw std::runtime_error("Somehow someway one of the delta arrays is not equivalent to its corresponding nabla array");

		for (int i = 0; i < nabla_b.size(); i++) {
			CMatrix t = add_cuda(nabla_b[i], delta_nabla_b[i]);
			nabla_b[i] = t;
		}
		for (int i = 0; i < nabla_w.size(); i++) {
			CMatrix t = add_cuda(nabla_w[i], delta_nabla_w[i]);
			nabla_w[i] = t;
		}
	}

	if (nabla_b.front().height != biasArray.front().height || nabla_b.front().width != biasArray.front().width ||
		nabla_w.front().height != weightsArray.front().height || nabla_w.front().width != weightsArray.front().width)
		throw std::runtime_error("Somehow someway one of the nabla arrays is not equivalent to its corresponding original array");

	const double scalar = learningRate / miniBatch.size();
	for (int i = 0; i < weightsArray.size(); i++) {
		weightsArray[i] = subtract_cuda(weightsArray[i], smultiply_cuda(nabla_w[i], scalar));
	}
	for (int i = 0; i < biasArray.size(); i++) {
		biasArray[i] = subtract_cuda(biasArray[i], smultiply_cuda(nabla_b[i], scalar));
	}
}

//Performs stochastic gradient descent to train the neural network
void NeuralNetwork::stochasticGradDescent(std::vector<std::pair<CMatrix, int>> trainingData, int epochs, int miniBatchSize, double learningRate, std::vector<std::pair<CMatrix, int>> testData) {
	std::random_device rd;
	std::mt19937 g(rd());

	int n = trainingData.size();
	for (int j = 0; j < epochs; j++) {

		std::shuffle(trainingData.begin(), trainingData.end(), g);

		std::vector<std::vector<std::pair<CMatrix, int>>> miniBatches;
		for (int k = 0; k < n; k += miniBatchSize) {
			int end = std::min(k + miniBatchSize, n);
			std::vector<std::pair<CMatrix, int>> miniBatch(trainingData.begin() + k, trainingData.begin() + end);
			miniBatches.push_back(miniBatch);
		}

		int count = 0;
		for (auto batch : miniBatches) {
			updateMiniBatch(batch, learningRate);
			std::cout << "Mini batch " << count << " of " << miniBatches.size() << " completed\n";
		}

		std::cout << "Epoch " << j << ": " << evaluate(testData) << " / " << testData.size() << "\n";
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