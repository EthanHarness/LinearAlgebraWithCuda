#include "CMatrix.cuh"
#include <chrono>
#include <ctime>

//Assume square matrices for now
__global__ void multiplyWithCuda(CMatrix A, CMatrix B, CMatrix C) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0;

	for (int i = 0; i < A.width; ++i) {
		sum += A.elements[row * A.width + i] * B.elements[i * B.width + col];
	}
	C.elements[row * C.width + col] = sum;

};

CMatrix multiply_cuda(CMatrix mat1, CMatrix mat2) {

	int rows = mat1.height;
	int cols = mat1.width;

	if (rows != mat2.height || cols != mat2.width || rows != cols)
		throw std::invalid_argument("Matrices are not the same size or matrices are not squares.");

	CMatrix res = createCMatrix(rows, cols);

	CMatrix device_matrix_A;
	device_matrix_A.width = mat1.width;
	device_matrix_A.height = mat1.height;
	size_t size = mat1.width * mat1.height * sizeof(double);
	cudaMalloc(&device_matrix_A.elements, size);
	cudaMemcpy(device_matrix_A.elements, mat1.elements, size, cudaMemcpyHostToDevice);

	CMatrix device_matrix_B;
	device_matrix_B.width = mat2.width;
	device_matrix_B.height = mat2.height;
	size = mat2.width * mat2.height * sizeof(double);
	cudaMalloc(&device_matrix_B.elements, size);
	cudaMemcpy(device_matrix_B.elements, mat2.elements, size, cudaMemcpyHostToDevice);

	CMatrix device_matrix_C;
	device_matrix_C.width = res.width;
	device_matrix_C.height = res.height;
	size = res.width * res.height * sizeof(double);
	cudaMalloc(&device_matrix_C.elements, size);
	cudaMemcpy(device_matrix_C.elements, res.elements, size, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(rows, cols);
	dim3 numBlocks(1, 1);
	multiplyWithCuda << <numBlocks, threadsPerBlock >> > (device_matrix_A, device_matrix_B, device_matrix_C);
	cudaDeviceSynchronize();

	cudaMemcpy(res.elements, device_matrix_C.elements, size, cudaMemcpyDeviceToHost);
	cudaFree(device_matrix_A.elements);
	cudaFree(device_matrix_B.elements);
	cudaFree(device_matrix_C.elements);

	return res;
};

int main()
{

    std::cout << "Mat1\n";

    CMatrix CMatrixObj = createCMatrix(5, 5);
    std::function<double(int, int)> foo1;
    foo1 = [](int x, int y) {
        return (double)((y * y) + x);
    };
    setCMatrix(foo1, CMatrixObj);
    printCMatrix(CMatrixObj);

    std::cout << "\nMat2\n";

    CMatrix CMatrixObj1 = createCMatrix(5, 5);
    std::function<double(int, int)> foo; 
    foo = [](int x, int y) {
        return (double)((x * x) + y);
    };
    setCMatrix(foo, CMatrixObj1);
    printCMatrix(CMatrixObj1);

    std::cout << "\nMat3\n";
	CMatrix CMatrixObj2 = CMatrixAdd(CMatrixObj, CMatrixObj1);
    printCMatrix(CMatrixObj2);

    std::cout << "\nMat4\n";

    std::function<double(int, int)> foo2;
    foo2 = [](int x, int y) {
        return (double)(x + 1);
    };
    CMatrix m1 = createCMatrix(2, 2);
    CMatrix m2 = createCMatrix(2, 2);
    setCMatrix(foo2, m1);
    setCMatrix(foo2, m2);

    CMatrix m3 = CMatrixMultiply(CMatrixObj1, CMatrixObj2);
    printCMatrix(m3);

    std::cout << "\nMat5\n";

    CMatrix m4 = multiply_cuda(CMatrixObj1, CMatrixObj2);
    printCMatrix(m4);

	CMatrix m5 = createCMatrix(5000, 5000);
	CMatrix m6 = createCMatrix(5000, 5000);

	std::function<double(int, int)> foo3;
	foo3 = [m5](int x, int y) {
		return (x * m5.width * y);
	};
	setCMatrix(foo3, m5);
	setCMatrix(foo3, m6);

	auto t_start1 = std::chrono::high_resolution_clock::now();
	CMatrix m7 = CMatrixMultiply(m5, m6);
	auto t_end1 = std::chrono::high_resolution_clock::now();

	auto t_start2 = std::chrono::high_resolution_clock::now();
	CMatrix m8 = multiply_cuda(m5, m6);
	auto t_end2 = std::chrono::high_resolution_clock::now();

	std::cout << "Time 1: " << std::chrono::duration<double, std::milli>(t_end1 - t_start1).count() << std::endl;
	std::cout << "Time 2: " << std::chrono::duration<double, std::milli>(t_end2 - t_start2).count() << std::endl;


	

    return 0;
}
