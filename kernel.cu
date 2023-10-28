#include "CMatrix.cuh"
#include <chrono>
#include <ctime>

#define ITERATIONS 1000

void CudaVNonCuda();


int main() {
    CudaVNonCuda();
    return 0;
}

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
    CMatrix CMatrixObj2 = CMatrixAdd(CMatrixObj, CMatrixObj1);

    //This does a bunch of Matrix multiplications.
    for(int i = 0; i < ITERATIONS; i+=10) {
        std::cout << "Iteration : " << i/10 << std::endl;
        std::cout << "Size of Matrix's are : " << i << "x" << i << std::endl;
        CMatrix m1 = createCMatrix(i, i);
        CMatrix m2 = createCMatrix(i, i);
        setCMatrix(foo, m1);
        setCMatrix(foo, m2);

        clock_t now = clock();
        CMatrix m3 = CMatrixMultiply(m1, m2);
        std:: cout << "TIME: " << clock() - now << std::endl;
            
        now = clock();
        CMatrix m4 = multiply_cuda(m1, m2);
        std::cout << "TIME : " << clock() - now << std::endl << std::endl;
   }
}