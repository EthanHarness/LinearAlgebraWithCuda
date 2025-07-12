#include "NeuralNetwork.cuh"

//takes in an array of ints and the size of the array
//each value in the array is the size of the corresponding hidden layer
//first value is the size of the input layer. We will not be setting anything for the first layer since its the input. 
NeuralNetwork::NeuralNetwork(int layers[], int size) {
	const double variance = 1;
	const ActivationFunctionE temporaryActivationFunctionConstant = ActivationFunctionE::Sigmoid;
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0, variance);

	std::function<double(int, int)> randomNumberGeneratorFunction;
	randomNumberGeneratorFunction = [generator, distribution](int x, int y) mutable {
		return distribution(generator);
	};


	for (int i = 1; i < size; i++) {
		CMatrix weights = createCMatrix(layers[i - 1], layers[i]);
		CMatrix bias = createCMatrix(1, layers[i]);
		setCMatrix(randomNumberGeneratorFunction, weights);
		setCMatrix(randomNumberGeneratorFunction, bias);

		weightsArray.push_back(weights);
		biasArray.push_back(bias);
		activationFunctions.push_back(temporaryActivationFunctionConstant);
	}

	networkSize = size - 1;
}

//Returns the output layer based off the network
CMatrix NeuralNetwork::processInput(CMatrix inputNodes) {
	using namespace std;
	CMatrix temp1, temp2, temp3, res;
	temp1 = inputNodes;
	for (int i = 0; i < networkSize; i++, temp1 = res) {

		temp2 = multiply_cuda(temp1, weightsArray[i]);
		temp3 = add_cuda(temp2, biasArray[i]);

		switch (activationFunctions[i]) {
			case ActivationFunctionE::Sigmoid:
				res = sigmoid_cuda(temp3);
				break;
			case ActivationFunctionE::Tanh:
				res = tanh_cuda(temp3);
				break;
			case ActivationFunctionE::Relu:
				res = relu_cuda(temp3);
				break;
			case ActivationFunctionE::Unknown:
				throw std::invalid_argument("Unknown activation function found at activationFunctions " + std::to_string(i));
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
void NeuralNetwork::stochasticGradDescent(std::vector<std::pair<CMatrix, int>> trainingData, int epochs, int miniBatchSize, double learningRate, std::vector<std::pair<CMatrix, int>> testData) {
	std::random_device rd;
	std::mt19937 g(rd());

	int n = trainingData.size();
	for(int j = 0; j < epochs; j++) {

		std::shuffle(trainingData.begin(), trainingData.end(), g);
		
		std::vector<std::vector<std::pair<CMatrix, int>>> miniBatches;
		for(int k = 0; k < n; k += miniBatchSize) {
			int end = std::min(k+miniBatchSize, n);
			std::vector<std::pair<CMatrix, int>> miniBatch(trainingData.begin() + k, trainingData.begin() + end);
			miniBatches.push_back(miniBatch);
		}

		for (auto batch : miniBatches) {
			updateMiniBatch(batch, learningRate);
		}

		std::cout << "Epoch " << j << ": " << evaluate(testData) << " / " << testData.size() << "\n";
	}
}

void NeuralNetwork::updateMiniBatch(std::vector<std::pair<CMatrix, int>> miniBatch, double learningRate) {
	std::function<double(int, int)> zeroFunc = [](int x, int y) {
		return 0;
	};

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
		nabla_b.push_back(t);
	}

	for (auto io : miniBatch) {
		CMatrix input = io.first;
		int expectedOutput = io.second;
		auto output = backprop(input, expectedOutput);
		std::vector<CMatrix> delta_nabla_b = output.first;
		std::vector<CMatrix> delta_nabla_w = output.second;

		if (nabla_b.front().height != delta_nabla_b.front().height || nabla_b.front().width != delta_nabla_b.front().width || 
			nabla_w.front().height != delta_nabla_w.front().height || nabla_w.front().width != delta_nabla_w.front().width)
			std::cerr << "Somehow someway one of the delta arrays is not equivalent to its corresponding nabla array\n";

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
			std::cerr << "Somehow someway one of the nabla arrays is not equivalent to its corresponding original array\n";

	const double scalar = learningRate/miniBatch.size();
	for (int i = 0; i < weightsArray.size(); i++) {
		weightsArray[i] = subtract_cuda(weightsArray[i], smultiply_cuda(nabla_w[i], scalar));
	}
	for (int i = 0; i < biasArray.size(); i++) {
		biasArray[i] = subtract_cuda(biasArray[i], smultiply_cuda(nabla_b[i], scalar));
	}
}

//WIP No progress done
std::pair<std::vector<CMatrix>, std::vector<CMatrix>> NeuralNetwork::backprop(CMatrix networkInput, int expectedInputsOutput) {
	std::vector<CMatrix> vec1; 
	vec1.push_back(createCMatrix(1,1));
	std::vector<CMatrix> vec2;
	vec2.push_back(createCMatrix(1,1));
	return std::pair<std::vector<CMatrix>, std::vector<CMatrix>>(vec1, vec2);
}

//WIP No progress done
int NeuralNetwork::evaluate(std::vector<std::pair<CMatrix, int>>) {
	return 0;
}

//Converts a string of our activation function to an enum ActivationFunctionE 
//Probably could refactor this to be more efficient
ActivationFunctionE NeuralNetwork::stringToActivationFunction(const std::string& str) {
    if (str == "sigmoid") return ActivationFunctionE::Sigmoid;
    if (str == "tanh") return ActivationFunctionE::Tanh;
    if (str == "relu") return ActivationFunctionE::Relu;
    return ActivationFunctionE::Unknown;
}