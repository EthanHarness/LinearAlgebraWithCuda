#include "CMatrix.cuh"
#include <chrono>
#include <ctime>

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
    CMatrix m1 = createCMatrix(2, 4);
    CMatrix m2 = createCMatrix(4, 2);
    setCMatrix(foo2, m1);
    setCMatrix(foo2, m2);

    CMatrix m3 = CMatrixMultiply(m1, m2);
    printCMatrix(m3);

    std::cout << "\nMat5\n";

    CMatrix m4 = multiply_cuda(m1, m2);
    printCMatrix(m4);

    return 0;
}
