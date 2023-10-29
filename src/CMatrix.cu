#include "CMatrix.cuh"

__global__ void multiplyWithCuda(CMatrix A, CMatrix B, CMatrix C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0;

	for (int i = 0; i < A.width; i++) {
		sum += A.elements[row * A.width + i] * B.elements[i * B.width + col];
	}
	C.elements[row * C.width + col] = sum;
};

__global__ void smultiplyWithCuda(CMatrix A, CMatrix B, double scalar) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	B.elements[row * B.width + col] = A.elements[row * A.width + col] * scalar;
};

__global__ void addWithCuda(CMatrix A, CMatrix B, CMatrix C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	C.elements[row * C.width + col] = A.elements[row * A.width + col] + B.elements[row * B.width + col];
};

__global__ void sigmoidWithCuda(CMatrix A, CMatrix B) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	B.elements[row * B.width + col] = 1 / (1 + exp(A.elements[row * A.width + col]));
}

__global__ void tanhWithCuda(CMatrix A, CMatrix B) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	B.elements[row * B.width + col] = atanh(A.elements[row + A.width + col]);
}

__global__ void reluWithCuda(CMatrix A, CMatrix B) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	B.elements[row * B.width + col] = max(A.elements[row + A.width + col], 0.0);
}

__global__ void squareDiffWithCuda(CMatrix A, CMatrix B, CMatrix C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double diff = A.elements[row * A.width + col] - B.elements[row * B.width + col];
	C.elements[row * C.width + col] = pow(diff, 2);
}

void setCMatrix(std::function<double(int, int)> func, CMatrix& res) {
	for (int i = 0; i < res.height; i++) {
		for (int j = 0; j < res.width; j++) {
			res.elements[i * res.width + j] = func(i, j);
		}
	}
}

void printCMatrix(const CMatrix& res) {
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

double getMax(CMatrix mat) {
	double max = mat.elements[0];
	for (int j = 1; j < mat.width; j++) {
		max = mat.elements[mat.width + j] > max ? mat.elements[mat.width + j] : max;
	}
	return max;
}

double getMax(CMatrix mat, int row) {
	double max = mat.elements[row * mat.width];
	for (int j = 1; j < mat.width; j++) {
		max = mat.elements[row * mat.width + j] > max ? mat.elements[row * mat.width + j] : max;
	}
	return max;
}

CMatrix multiply_cuda(CMatrix mat1, CMatrix mat2) {
	int row1 = mat1.height;
	int row2 = mat2.height;
	int col1 = mat1.width;
	int col2 = mat2.width;

	if (col1 != row2)
		throw std::invalid_argument("Columns of matrix 1 do not equal the rows of matrix 2.");

	CMatrix res = createCMatrix(row1, col2);

	CMatrix device_matrix_A;
	CMatrix device_matrix_B;
	CMatrix device_matrix_C;

	device_matrix_A.width = mat1.width;device_matrix_A.height = mat1.height;
	device_matrix_B.width = mat2.width;device_matrix_B.height = mat2.height;
	device_matrix_C.width = res.width;device_matrix_C.height = res.height;

	size_t size_A = mat1.width * mat1.height * sizeof(double);
	size_t size_B = mat2.width * mat2.height * sizeof(double);
	size_t size_C = res.width * res.height * sizeof(double);

	cudaMalloc(&device_matrix_A.elements, size_A);
	cudaMemcpy(device_matrix_A.elements, mat1.elements, size_A, cudaMemcpyHostToDevice);
	cudaMalloc(&device_matrix_B.elements, size_B);
	cudaMemcpy(device_matrix_B.elements, mat2.elements, size_B, cudaMemcpyHostToDevice);
	cudaMalloc(&device_matrix_C.elements, size_C);
	cudaMemcpy(device_matrix_C.elements, res.elements, size_C, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(col2, row1);
	dim3 numBlocks(1, 1);
	multiplyWithCuda <<<numBlocks, threadsPerBlock >>> (device_matrix_A, device_matrix_B, device_matrix_C);
	cudaDeviceSynchronize();

	cudaMemcpy(res.elements, device_matrix_C.elements, size_C, cudaMemcpyDeviceToHost);
	cudaFree(device_matrix_A.elements);
	cudaFree(device_matrix_B.elements);
	cudaFree(device_matrix_C.elements); 
	
	return res;
};

CMatrix smultiply_cuda(CMatrix mat, double scalar) {
	int rows = mat.height;
	int cols = mat.width;

	CMatrix res = createCMatrix(rows, cols);

	CMatrix device_matrix_A;
	CMatrix device_matrix_B;

	device_matrix_A.width = mat.width;device_matrix_A.height = mat.height;
	device_matrix_B.width = res.width;device_matrix_B.height = res.height;

	size_t size_A = mat.width * mat.height * sizeof(double);
	size_t size_B = res.width * res.height * sizeof(double);

	cudaMalloc(&device_matrix_A.elements, size_A);
	cudaMemcpy(device_matrix_A.elements, mat.elements, size_A, cudaMemcpyHostToDevice);
	cudaMalloc(&device_matrix_B.elements, size_B);
	cudaMemcpy(device_matrix_B.elements, res.elements, size_B, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(cols, rows);
	dim3 numBlocks(1, 1);
	smultiplyWithCuda <<<numBlocks, threadsPerBlock >>> (device_matrix_A, device_matrix_B, scalar);
	cudaDeviceSynchronize();

	cudaMemcpy(res.elements, device_matrix_B.elements, size_B, cudaMemcpyDeviceToHost);
	cudaFree(device_matrix_A.elements);
	cudaFree(device_matrix_B.elements);

	return res;
}

CMatrix add_cuda(CMatrix mat1, CMatrix mat2) {
	int row1 = mat1.height;
	int row2 = mat2.height;
	int col1 = mat1.width;
	int col2 = mat2.width;

	if (col1 != col2 && row1 != row2)
		throw std::invalid_argument("Columns/rows of matrix 1 do not equal the columns/rows of matrix 2.");

	CMatrix res = createCMatrix(row1, col1);

	CMatrix device_matrix_A;
	CMatrix device_matrix_B;
	CMatrix device_matrix_C;
	
	device_matrix_A.width = mat1.width;device_matrix_A.height = mat1.height;
	device_matrix_B.width = mat2.width;device_matrix_B.height = mat2.height;
	device_matrix_C.width = res.width;device_matrix_C.height = res.height;
	
	size_t size_A = mat1.width * mat1.height * sizeof(double);
	size_t size_B = mat2.width * mat2.height * sizeof(double);
	size_t size_C = res.width * res.height * sizeof(double);

	cudaMalloc(&device_matrix_A.elements, size_A);
	cudaMemcpy(device_matrix_A.elements, mat1.elements, size_A, cudaMemcpyHostToDevice);
	cudaMalloc(&device_matrix_B.elements, size_B);
	cudaMemcpy(device_matrix_B.elements, mat2.elements, size_B, cudaMemcpyHostToDevice);
	cudaMalloc(&device_matrix_C.elements, size_C);
	cudaMemcpy(device_matrix_C.elements, res.elements, size_C, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(col2, row1);
	dim3 numBlocks(1, 1);
	addWithCuda <<<numBlocks, threadsPerBlock >>> (device_matrix_A, device_matrix_B, device_matrix_C);
	cudaDeviceSynchronize();

	cudaMemcpy(res.elements, device_matrix_C.elements, size_C, cudaMemcpyDeviceToHost);
	cudaFree(device_matrix_A.elements);
	cudaFree(device_matrix_B.elements);
	cudaFree(device_matrix_C.elements);

	return res;
}

CMatrix sigmoid_cuda(CMatrix mat1) {
	int row = mat1.height;
	int col = mat1.width;

	CMatrix res = createCMatrix(row, col);

	CMatrix device_matrix_A;
	CMatrix device_matrix_B;

	device_matrix_A.width = mat1.width;device_matrix_A.height = mat1.height;
	device_matrix_B.width = res.width;device_matrix_B.height = res.height;

	size_t size_A = mat1.width * mat1.height * sizeof(double);
	size_t size_B = res.width * res.height * sizeof(double);

	cudaMalloc(&device_matrix_A.elements, size_A);
	cudaMemcpy(device_matrix_A.elements, mat1.elements, size_A, cudaMemcpyHostToDevice);
	cudaMalloc(&device_matrix_B.elements, size_B);
	cudaMemcpy(device_matrix_B.elements, res.elements, size_B, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(col, row);
	dim3 numBlocks(1, 1);
	sigmoidWithCuda <<<numBlocks, threadsPerBlock >>> (device_matrix_A, device_matrix_B);
	cudaDeviceSynchronize();

	cudaMemcpy(res.elements, device_matrix_B.elements, size_B, cudaMemcpyDeviceToHost);
	cudaFree(device_matrix_A.elements);
	cudaFree(device_matrix_B.elements);

	return res;
}

CMatrix relu_cuda(CMatrix mat1) {
	int row = mat1.height;
	int col = mat1.width;

	CMatrix res = createCMatrix(row, col);

	CMatrix device_matrix_A;
	CMatrix device_matrix_B;
	
	device_matrix_A.width = mat1.width;device_matrix_A.height = mat1.height;
	device_matrix_B.width = res.width;device_matrix_B.height = res.height;

	size_t size_A = mat1.width * mat1.height * sizeof(double);
	size_t size_B = res.width * res.height * sizeof(double);

	cudaMalloc(&device_matrix_A.elements, size_A);
	cudaMemcpy(device_matrix_A.elements, mat1.elements, size_A, cudaMemcpyHostToDevice);
	cudaMalloc(&device_matrix_B.elements, size_B);
	cudaMemcpy(device_matrix_B.elements, res.elements, size_B, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(col, row);
	dim3 numBlocks(1, 1);
	reluWithCuda <<<numBlocks, threadsPerBlock >>> (device_matrix_A, device_matrix_B);
	cudaDeviceSynchronize();

	cudaMemcpy(res.elements, device_matrix_B.elements, size_B, cudaMemcpyDeviceToHost);
	cudaFree(device_matrix_A.elements);
	cudaFree(device_matrix_B.elements);

	return res;
}

CMatrix tanh_cuda(CMatrix mat1) {
	int row = mat1.height;
	int col = mat1.width;

	CMatrix res = createCMatrix(row, col);

	CMatrix device_matrix_A;
	CMatrix device_matrix_B;
	
	device_matrix_A.width = mat1.width;device_matrix_A.height = mat1.height;
	device_matrix_B.width = res.width;device_matrix_B.height = res.height;

	size_t size_A = mat1.width * mat1.height * sizeof(double);
	size_t size_B = res.width * res.height * sizeof(double);

	cudaMalloc(&device_matrix_A.elements, size_A);
	cudaMemcpy(device_matrix_A.elements, mat1.elements, size_A, cudaMemcpyHostToDevice);
	cudaMalloc(&device_matrix_B.elements, size_B);
	cudaMemcpy(device_matrix_B.elements, res.elements, size_B, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(col, row);
	dim3 numBlocks(1, 1);
	tanhWithCuda <<<numBlocks, threadsPerBlock >>> (device_matrix_A, device_matrix_B);
	cudaDeviceSynchronize();

	cudaMemcpy(res.elements, device_matrix_B.elements, size_B, cudaMemcpyDeviceToHost);
	cudaFree(device_matrix_A.elements);
	cudaFree(device_matrix_B.elements);

	return res;
}

CMatrix computeLossMatrix_cuda(CMatrix computedMatrix, CMatrix expectedMatrix) {
	int row = computedMatrix.height;
	int col = computedMatrix.width;

	if (expectedMatrix.height != row && expectedMatrix.width != col)
		throw std::invalid_argument("Matricies are not of equal size");

	CMatrix res = createCMatrix(row, col);

	CMatrix device_matrix_A;
	CMatrix device_matrix_B;
	CMatrix device_matrix_C;

	device_matrix_A.width = computedMatrix.width;device_matrix_A.height = computedMatrix.height;
	device_matrix_B.width = expectedMatrix.width;device_matrix_B.height = expectedMatrix.height;
	device_matrix_C.width = res.width;device_matrix_C.height = res.height;

	size_t size_A = computedMatrix.width * computedMatrix.height * sizeof(double);
	size_t size_B = expectedMatrix.width * expectedMatrix.height * sizeof(double);
	size_t size_C = res.width * res.height * sizeof(double);

	cudaMalloc(&device_matrix_A.elements, size_A);
	cudaMemcpy(device_matrix_A.elements, computedMatrix.elements, size_A, cudaMemcpyHostToDevice);
	cudaMalloc(&device_matrix_B.elements, size_B);
	cudaMemcpy(device_matrix_B.elements, expectedMatrix.elements, size_B, cudaMemcpyHostToDevice);
	cudaMalloc(&device_matrix_C.elements, size_C);
	cudaMemcpy(device_matrix_C.elements, res.elements, size_C, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(col, row);
	dim3 numBlocks(1, 1);
	squareDiffWithCuda <<<numBlocks, threadsPerBlock >>> (device_matrix_A, device_matrix_B, device_matrix_C);
	cudaDeviceSynchronize();

	cudaMemcpy(res.elements, device_matrix_C.elements, size_C, cudaMemcpyDeviceToHost);
	cudaFree(device_matrix_A.elements);
	cudaFree(device_matrix_B.elements);
	cudaFree(device_matrix_C.elements);

	return res;
}
