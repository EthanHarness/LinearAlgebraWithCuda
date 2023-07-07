#include "CMatrix.cuh"

int main()
{
    int** standardArr = (int**)malloc(3 * sizeof(int*));
    for (int i = 0; i < 3; i++) {
        standardArr[i] = (int*)malloc(4 * sizeof(int));
        standardArr[i][0] = (i * 3) + 1;
        standardArr[i][1] = (i * 3) + 2;
        standardArr[i][2] = (i * 3) + 3;
        standardArr[i][3] = (i * 3) + 4;
    }

    CMatrix<int> CMatrixObj = CMatrix<int>(3, 4);
    CMatrixObj.setCMatrix(standardArr);
    CMatrixObj.printCMatrix();

    return 0;
}
