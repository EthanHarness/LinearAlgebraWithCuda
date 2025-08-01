﻿#include "NeuralNetwork.h"
#include "CMatrix.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void helperFunction(NeuralNetwork network, CMatrix inputNodes);
void CudaVNonCuda();
std::vector<std::pair<CMatrix, int>> readTestData();
std::vector<std::pair<CMatrix, int>> readTrainingData();

int main() {
    //Takes in mnist training sets
    std::vector<std::pair<CMatrix, int>> testData = readTestData();
    std::vector<std::pair<CMatrix, int>> trainData = readTrainingData();
    std::cout << "Train Data samples: " << trainData.size() << "\n";
    std::cout << "Test Data samples: " << testData.size() << "\n";

    /*
    Initializes network strucutre. 
    784 (28x28) neurons in layer 1 with 784x10 weights total (CUDA IS GOATED AT THIS)
    10 neurons layer 2 with again 10x10 weights total
    Final layer holds output information. We are choosing 1 out of 10 outputs
    */
    int layer1 = testData[0].first.height;
    const int layer2 = 16;
    const int layer3 = 12;
    const int layer4 = 10;
    const int networkSize = 4;
    const int epochs = 100;
    const int batchSize = 3000;
    const double learningRate = .05;
    int networkStructure[] = {layer1, layer2, layer3, layer4};
    NeuralNetwork network = NeuralNetwork(networkStructure, networkSize);
    network.stochasticGradDescent(trainData, epochs, batchSize, learningRate, testData);
    //CudaVNonCuda();
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

    const int limit = 1000;
    int count = 0;
    while (std::getline(file, line) && (count < limit || limit == -1)) {
        count++;
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

    const int limit = 9000;
    int count = 0;
    while (std::getline(file, line) && (count < limit || limit == -1)) {
        count++;
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