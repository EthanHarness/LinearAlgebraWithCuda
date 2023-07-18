#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <functional>
#include <stdexcept>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct {
	int width;
	int height;
	double* elements;
} CMatrix;

void setCMatrix(std::function<double(int, int)> func, CMatrix res) {
	for (int i = 0; i < res.height; i++) {
		for (int j = 0; j < res.width; j++) {
			res.elements[i * res.width + j] = func(i, j);
		}
	}
}

void printCMatrix(CMatrix res) {
	for (int i = 0; i < res.height; i++) {
		for (int j = 0; j < res.width; j++) {
			std::cout << res.elements[i * res.width + j] << " ";
		}
		std::cout << "\n";
	}
}

CMatrix createCMatrix(int rows, int cols) {
	CMatrix res;
	res.height = rows;
	res.width = cols;
	res.elements = (double*)calloc(rows * cols, sizeof(double));
	return res;
}

CMatrix CMatrixAdd(CMatrix mat1, CMatrix mat2) {
	int cols = mat1.width;
	int rows = mat2.height;

	if (cols != mat2.width || rows != mat2.height)
		throw std::invalid_argument("Matricies are not the same size.");

	CMatrix result = createCMatrix(cols, rows);
	std::function<double(int, int)> add = [mat1, mat2, cols](int i, int j) {
		return mat1.elements[i * cols + j] + mat2.elements[i * cols + j];
	};

	setCMatrix(add, result);
	return result;

}

CMatrix CMatrixSMultiply(CMatrix mat, double scalar) {
	int cols = mat.width;
	int rows = mat.height;

	CMatrix res = createCMatrix(cols, rows);
	std::function<double(int, int)> smult = [mat, scalar](int i, int j) {
		return mat.elements[i * mat.width + j] * scalar;
	};
	setCMatrix(smult, res);
	return res;
}

CMatrix CMatrixMultiply(CMatrix mat1, CMatrix mat2) {
	int row1 = mat1.height;
	int row2 = mat2.height;
	int col1 = mat1.width;
	int col2 = mat2.width;

	if (col1 != row2)
		throw std::invalid_argument("Columns of matrix 1 do not equal the rows of matrix 2.");

	CMatrix res = createCMatrix(row1, col2);
	double* resHead = res.elements;
	double* mat1Head = mat1.elements;
	double* mat2Head = mat2.elements;

	for (int i = 0; i < row1; i++) {
		for (int j = 0; j < col2; j++) {
			for (int k = 0; k < row2; k++) {
				resHead[i * res.width + j] += mat1Head[i * col1 + k] * mat2Head[k * col2 + j];
			}
		}
	}

	return res;
}
