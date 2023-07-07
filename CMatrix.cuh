#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename T>
class CMatrix {
private:
	int numRows;
	int numCols;
	int** matHead;

public:
	CMatrix(int rows, int cols) {
		numRows = rows;
		numCols = cols;
		matHead = (T**)calloc(rows, sizeof(T*));
		for (int i = 0; i < rows; i++) {
			matHead[i] = (T*)calloc(cols, sizeof(T));
		}
	}

	void setCMatrix(T** arr) {
		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols ; j++) {
				matHead[i][j] = arr[i][j];
			}
		}
	}

 	void printCMatrix() {
		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {
				printf("%d ", matHead[i][j]);;
			}
			printf("\n");
		}
	}

	int getRows() {
		return numRows;
	}

	int getCols() {
		return numCols;
	}

};

