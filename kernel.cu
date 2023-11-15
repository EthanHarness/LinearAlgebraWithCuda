#include "NeuralNetwork.cuh"
#include <chrono>
#include <ctime>

#define ITERATIONS 10

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
    int networkStructure[] = {784, 10, 10};
    NeuralNetwork network = NeuralNetwork(networkStructure, 3);
    //CMatrix outputLayer = network.processInput(testData[0].first);
    helperFunction(network, testData[0].first);

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
    setCMatrix(foo2, dummyWeights);
    setCMatrix(foo3, dummyBias);

    std::cout << "Network 1" << "\n";
    printCMatrix(dummyInput);
    std::cout << "Network 2" << "\n";
    printCMatrix(dummyWeights);
    std::cout << "Network 3" << "\n";
    printCMatrix(dummyBias);


    CMatrix res = multiply_cuda(dummyInput, dummyWeights);
    std::cout << "Network 4" << std::endl;
    printCMatrix(res);

    res = add_cuda(res, dummyBias);
    std::cout << "Network 5" << std::endl;
    printCMatrix(res);
}

//A little DEMO function I wrote to compare the speed of 
//CUDA matrix multiplication vs regular matrix multiplication
void CudaVNonCuda() {
    //Creates and sets a bunch of CMatrix's (Mainly for testing purposes)
    CMatrix CMatrixObj = createCMatrix(5, 5);
    CMatrix CMatrixObj1 = createCMatrix(5, 5);
    std::function<double(int, int)> foo; 
    std::function<double(int, int)> foo1;
    std::function<double(int, int)> foo2;

    foo = [](int x, int y) {
        return (double)((x * x) + y);
    };
    foo1 = [](int x, int y) {
        return (double)((y * y) + x);
    };
    foo2 = [](int x, int y) {
        return (double)(x + 1);
    };

    setCMatrix(foo1, CMatrixObj);
    setCMatrix(foo, CMatrixObj1);

    std::cout << "Mat 1" << std::endl;
    printCMatrix(CMatrixObj);
    std::cout << "Mat 2" << std::endl;
    printCMatrix(CMatrixObj1);

    CMatrix CMatrixObj2 = multiply_cuda(CMatrixObj, CMatrixObj1);

    std::cout << "Mat 3" << std::endl;
    printCMatrix(CMatrixObj2);


    //This does a bunch of Matrix multiplications.
    for(int i = 0; i < ITERATIONS*10; i+=10) {
        std::cout << "Iteration : " << i/10 << std::endl;
        std::cout << "Size of Matrix's are : " << i << "x" << i << std::endl;
        CMatrix m1 = createCMatrix(i, i);
        CMatrix m2 = createCMatrix(i, i);
        setCMatrix(foo, m1);
        setCMatrix(foo, m2);

        clock_t now = clock();
        CMatrix m3 = CMatrixMultiply(m1, m2);
        std::cout << "TIME: " << clock() - now << std::endl;
            
        now = clock();
        CMatrix m4 = multiply_cuda(m1, m2);
        std::cout << "TIME : " << clock() - now << std::endl << std::endl;
   }
}

//Reads in test data for our NN to store
std::vector<std::pair<CMatrix, int>> readTestData() {
    std::ifstream file("data/mnist_test.csv");

    if (!file.is_open()) {
        std::cerr << "Error: File could not be opened." << std::endl;
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
        row.erase(row.begin());

        CMatrix testingDataCMatrix = createCMatrix(1, 784);
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
        std::cerr << "Error: File could not be opened." << std::endl;
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
        row.erase(row.begin());

        CMatrix testingDataCMatrix = createCMatrix(1, 784);
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