#include "NeuralNetwork.cuh"
#include "CMatrix.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <stdexcept>

void helperFunction(NeuralNetwork network, CMatrix inputNodes);
void CudaVNonCuda();
std::vector<std::pair<CMatrix, int>> readTestData();
std::vector<std::pair<CMatrix, int>> readTrainingData();

int main() {
    //CudaVNonCuda();

    //Takes in mnist training sets
    std::vector<std::pair<CMatrix, int>> testData = readTestData();
    std::vector<std::pair<CMatrix, int>> trainData = readTrainingData();

    /*
    Initializes network strucutre. 
    784 (28x28) neurons in layer 1 with 784x10 weights total (CUDA IS GOATED AT THIS)
    10 neurons layer 2 with again 10x10 weights total
    Final layer holds output information. We are choosing 1 out of 10 outputs
    */
    const int layer1 = 784;
    const int layer2 = 10;
    const int layer3 = 10;
    int networkStructure[] = {layer1, layer2, layer3};
    NeuralNetwork network = NeuralNetwork(networkStructure, 3);
    

    //helperFunction(network, testData[0].first);

    CMatrix outputLayer = network.processInput(testData[0].first);
    CMatrix knownMatrix = createCMatrix(1, 10);
    knownMatrix.elements[testData[0].second] = 1;
    CMatrix error = computeLossMatrix_cuda(outputLayer, knownMatrix);

    std::cout << "Known Matrix\n";
    printCMatrix(knownMatrix);
    std::cout << "\n";

    std::cout << "Predicted\n";
    printCMatrix(outputLayer);
    std::cout << "\n";

    std::cout << "Squared Diff(Error)\n";
    printCMatrix(error);
    std::cout << "\n";



    return 0;
}

//This function is to test various functions. 
//Probably should have written unit tests but fuck it.
void helperFunction(NeuralNetwork network, CMatrix inputNodes) {

    CMatrix dummyInput = createCMatrix(1, 5);
    CMatrix dummyWeights = createCMatrix(5, 3);
    CMatrix dummyBias = createCMatrix(1, 3);

    std::function<double(int, int)> foo1; 
    std::function<double(int, int)> foo2;
    std::function<double(int, int)> foo3;

    foo1 = [](int x, int y) {
        return (double)(x + 1.14 + y) * 1.34;
    };
    foo2 = [](int x, int y) {
        return (double)(x - 2.82 + y) * 8.21;
    };
    foo3 = [](int x, int y) {
        return (double)(x + 3.91 - y) * 8.92;
    };

    setCMatrix(foo1, dummyInput);
    CMatrix res = relu_cuda(dummyInput);
    printCMatrix(dummyInput);
    std::cout << "\n";
    printCMatrix(res);
}

//A little DEMO function I wrote to compare the speed of 
//CUDA matrix multiplication vs regular matrix multiplication
void CudaVNonCuda() {
    //Creates and sets a bunch of CMatrix's (Mainly for testing purposes)
    const int iterations = 10;
    const int matrix_scale_factor = 8;

    std::function<double(int, int)> foo = [](int x, int y) {
        return (double)((x + 2*y));
    };

    //This does a bunch of Matrix multiplications.
    for(int i = 1; i < iterations*matrix_scale_factor; i+=matrix_scale_factor) {
        CMatrix m1 = createCMatrix(i+1, i);
        CMatrix m2 = createCMatrix(i, i+1);
        setCMatrix(foo, m1);
        setCMatrix(foo, m2);
        std::cout << "Size of Matrix 1 is : " << m1.height << "x" << m1.width << std::endl;
        std::cout << "Size of Matrix 2 is : " << m2.height << "x" << m2.width << std::endl;

        clock_t now = clock();
        CMatrix m3 = CMatrixMultiply(m1, m2);
        std::cout << "TIME Normal Mult: " << clock() - now << std::endl;
            
        now = clock();
        CMatrix m4 = multiply_cuda(m1, m2);
        std::cout << "TIME CUDA Mult : " << clock() - now << std::endl << std::endl;

        printCMatrix(m1);
        printCMatrix(m2);

        freeCMatrix(m1);
        freeCMatrix(m2);
        freeCMatrix(m3);
        freeCMatrix(m4);
   }
}

//Reads in test data for our NN to store
std::vector<std::pair<CMatrix, int>> readTestData() {
    std::ifstream file("data/mnist_test.csv");

    if (!file.is_open()) {
        throw std::runtime_error("Error: File could not be opened.");
    }

    std::string line;
    std::vector<std::pair<CMatrix, int>> testData;
    
    //Need to consume the first line since its just header information
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::vector<int> row;
        while (std::getline(ss, value, ',')) {
            int intValue = std::stoi(value);
            row.push_back(intValue);
        }

        int firstValue = row[0];
        CMatrix testingDataCMatrix = createCMatrix(row.size()-1, 1);
        row.erase(row.begin());
        
        std::function<double(int, int)> foo;
        foo = [row](int x, int y) {
            return (double)(row[y]);
        };
        setCMatrix(foo, testingDataCMatrix);

        testData.push_back(std::make_pair(testingDataCMatrix, firstValue));
    }

    file.close();
    return testData;
}

//Reads in training data for our NN to store
std::vector<std::pair<CMatrix, int>> readTrainingData() {
    std::ifstream file("data/mnist_train.csv");

    if (!file.is_open()) {
        throw std::runtime_error("Error: File could not be opened.");
    }

    std::string line;
    std::vector<std::pair<CMatrix, int>> testData;
    
    //Need to consume the first line since its just header information
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::vector<int> row;
        while (std::getline(ss, value, ',')) {
            int intValue = std::stoi(value);
            row.push_back(intValue);
        }

        int firstValue = row[0];
        CMatrix testingDataCMatrix = createCMatrix(row.size()-1, 1);
        row.erase(row.begin());

        std::function<double(int, int)> foo;
        foo = [row](int x, int y) {
            return (double)(row[y]);
        };
        setCMatrix(foo, testingDataCMatrix);

        testData.push_back(std::make_pair(testingDataCMatrix, firstValue));
    }

    file.close();
    return testData;
}