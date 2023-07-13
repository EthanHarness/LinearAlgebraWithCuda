#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <functional>
#include <stdexcept>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CMatrix {
private:
	int numRows;
	int numCols;
	double** matHead;

public:
	CMatrix(int rows, int cols) {
		numRows = rows;
		numCols = cols;
		matHead = (double**)calloc(rows, sizeof(double*));
		for (int i = 0; i < rows; i++) {
			matHead[i] = (double*)calloc(cols, sizeof(double));
		}
	}

	void setCMatrix(std::function<double(int, int)> func) {
		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {
				matHead[i][j] = func(i,j);
			}
		}
	}

	void setCMatrix(double** arr) {
		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols ; j++) {
				matHead[i][j] = arr[i][j];
			}
		}
	}

 	void printCMatrix() {
		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {
				std::cout << matHead[i][j] << " ";
			}
			std::cout << "\n";
		}
	}

	int getRows() {
		return numRows;
	}

	int getCols() {
		return numCols;
	}

	double** getHead() const {
		return matHead;
	}

	static CMatrix add(CMatrix mat1, CMatrix mat2) {

		int cols = mat1.getCols();
		int rows = mat2.getRows();

		if (cols != mat2.getCols() || rows != mat2.getRows())
			throw std::invalid_argument("Matricies are not the same size.");

		CMatrix result = CMatrix(cols, rows);
		std::function<double(int, int)> add = [mat1, mat2](int i, int j) {
			return mat1.getHead()[i][j] + mat2.getHead()[i][j];
		};

		result.setCMatrix(add);
		return result;

	}

	static CMatrix smultiply(CMatrix mat, double scalar) {
		int cols = mat.getCols();
		int rows = mat.getRows();

		CMatrix res = CMatrix(cols, rows);
		std::function<double(int, int)> smult = [mat, scalar](int i, int j) {
			return mat.getHead()[i][j] * scalar;
		};
		res.setCMatrix(smult);
		return res;
	}


	static CMatrix multiply(CMatrix mat1, CMatrix mat2) {
		int row1 = mat1.getRows();
		int row2 = mat2.getRows();
		int col1 = mat1.getCols();
		int col2 = mat2.getCols();

		if (col1 != row2)
			throw std::invalid_argument("Columns of matrix 1 do not equal the rows of matrix 2.");

		CMatrix res = CMatrix(row1, col2);
		double** resHead = res.getHead();
		double** mat1Head = mat1.getHead();
		double** mat2Head = mat2.getHead();

		for (int i = 0; i < row1; i++) {
			for (int j = 0; j < col2; j++) {
				for (int k = 0; k < row2; k++) {
					resHead[i][j] += mat1Head[i][k] * mat2Head[k][j];
				}
			}
		}

		return res;
	}

};

