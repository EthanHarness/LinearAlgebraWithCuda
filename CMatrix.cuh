#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <functional>
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


typedef struct {
    int width;
    int height;
    double* elements;
} CMatrix;

void setCMatrix(std::function<double(int, int)> func, CMatrix res);
void printCMatrix(CMatrix res);
CMatrix createCMatrix(int rows, int cols);
CMatrix CMatrixAdd(CMatrix mat1, CMatrix mat2);
CMatrix CMatrixSMultiply(CMatrix mat, double scalar);
CMatrix CMatrixMultiply(CMatrix mat1, CMatrix mat2);

CMatrix multiply_cuda(CMatrix mat1, CMatrix mat2);
CMatrix smultiply_cuda(CMatrix mat, double scalar);

double sigmoidFunction(double x);
double tanhFunction(double x);
double relu(double x);

