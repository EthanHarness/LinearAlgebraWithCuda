#include "CMatrix.cuh"

int main()
{

    std::cout << "Mat1\n";

    CMatrix CMatrixObj = CMatrix(5, 5);
    std::function<double(int, int)> foo1;
    foo1 = [](int x, int y) {
        return (double)((y * y) + x);
    };
    CMatrixObj.setCMatrix(foo1);
    CMatrixObj.printCMatrix();

    std::cout << "\nMat2\n";

    CMatrix CMatrixObj1 = CMatrix(5, 5);
    std::function<double(int, int)> foo; 
    foo = [](int x, int y) {
        return (double)((x * x) + y);
    };
    CMatrixObj1.setCMatrix(foo);
    CMatrixObj1.printCMatrix();

    std::cout << "\nMat3\n";
    CMatrix CMatrixObj2 = CMatrix::add(CMatrixObj1, CMatrixObj);
    CMatrixObj2.printCMatrix();

    std::cout << "\nMat4\n";

    std::function<double(int, int)> foo2;
    foo2 = [](int x, int y) {
        return (double)(x + 1);
    };
    CMatrix m1 = CMatrix(2, 2);
    CMatrix m2 = CMatrix(2, 2);
    m1.setCMatrix(foo2);
    m2.setCMatrix(foo2);

    CMatrix m3 = CMatrix::multiply(CMatrixObj1, CMatrixObj2);
    m3.printCMatrix();





    return 0;
}
