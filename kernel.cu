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

    std::function<double(int, int)> foo2;
    foo2 = [](int x, int y) {
        return (double)(x + 1);
    };


    for(int i = 0; i < 1000; i+=10) {
	std::cout << "Iteration : " << i << std::endl;
	CMatrix m1 = createCMatrix(i, i);
	CMatrix m2 = createCMatrix(i, i);
	setCMatrix(foo, m1);
	setCMatrix(foo, m2);

	clock_t now = clock();
	CMatrix m3 = CMatrixMultiply(m1, m2);
	//printCMatrix(m3);
	std:: cout << "TIME: " << clock() - now << std::endl;
	    
	now = clock();
	CMatrix m4 = multiply_cuda(m1, m2);
	//printCMatrix(m4);
	std::cout << "TIME : " << clock() - now << std::endl << std::endl;
   }

	
    return 0;
}
