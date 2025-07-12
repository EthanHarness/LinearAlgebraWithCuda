#pragma once 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdexcept>
#include <iostream>
#include <stdio.h>
#include <functional>

typedef struct {
    int width;
    int height;
    double* elements;
} CMatrix;

void setCMatrix(std::function<double(int, int)> func, CMatrix& res);
void printCMatrix(const CMatrix& res);
CMatrix createCMatrix(int rows, int cols);
CMatrix CMatrixAdd(CMatrix mat1, CMatrix mat2);
CMatrix CMatrixSMultiply(CMatrix mat, double scalar);
CMatrix CMatrixMultiply(CMatrix mat1, CMatrix mat2);
double getMax(CMatrix mat);
double getMax(CMatrix mat, int row);
void freeCMatrix(CMatrix &matrix);

CMatrix multiply_cuda(CMatrix mat1, CMatrix mat2);
CMatrix smultiply_cuda(CMatrix mat, double scalar);
CMatrix add_cuda(CMatrix mat1, CMatrix mat2);
CMatrix sigmoid_cuda(CMatrix mat1);
CMatrix sigmoid_prime_cuda(CMatrix mat1);
CMatrix tanh_cuda(CMatrix mat1);
CMatrix relu_cuda(CMatrix mat1);
CMatrix computeLossMatrix_cuda(CMatrix computedMatrix, CMatrix expectedMatrix);