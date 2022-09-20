#include "matrix.cuh"

__host__ Matrix* Matrix::GMatrixTerm(Matrix* matrix, Matrix* matrix_T, Matrix* weights) {
    Matrix* transpose_weights = Matrix::MultiplyGPU(matrix_T, weights);
    Matrix* multiply = Matrix::MultiplyGPU(transpose_weights, weights);

    return Matrix::MultiplyGPU(multiply, matrix);
}

__host__ Matrix* Matrix::Weights(Matrix* matrix) {
    vector<float> arr;
    for (size_t i = 0; i < matrix->rows; i++) {
        float* ptr = matrix->Row(i);
        arr.push_back(sqrt(pow(ptr[0], 2) + pow(ptr[1], 2) + pow(ptr[2], 2)));
        free(ptr);
    }

    size_t max_side_length = matrix->rows > matrix->columns ? matrix->rows : matrix->columns;

    Matrix* diagonal = Matrix::DiagonalMatrix(arr.data(), max_side_length, max_side_length);
    *diagonal = diagonal->Reciprocal();

    diagonal->DeviceTransfer(diagonal->device_alloc, diagonal);

    return diagonal;
}

__host__ vector<Matrix*> Matrix::WeightedLeastSquares(Matrix* matrix) {
    vector<Matrix*> matrices;
    matrices.push_back(matrix);

    Matrix::PopulateRandomHost(matrix, 0.0f, 1.0f);

    Matrix* transpose = Matrix::TransposeGPU(matrix);
    matrices.push_back(transpose);

    Matrix* weights = Matrix::Weights(matrix);
    matrices.push_back(weights);

    Matrix* G_term = Matrix::GMatrixTerm(matrix, transpose, weights);
    matrices.push_back(G_term);

    Matrix* inverse = Matrix::Create(G_term->rows, G_term->columns);
    Matrix::Inverse(*G_term, *inverse);

    inverse->DeviceTransfer(inverse->device_alloc, inverse);
    matrices.push_back(inverse);

    Matrix* multiply_t = Matrix::MultiplyGPU(inverse, transpose);
    matrices.push_back(multiply_t);

    Matrix* weight_multiply = Matrix::MultiplyGPU(multiply_t, weights);
    matrices.push_back(weight_multiply);

    Matrix* gradient = Matrix::MultiplyGPU(weight_multiply, weights);
    matrices.push_back(gradient);

    RandomFloat random(100.0f, 500.0f, 1);

    Matrix* delta = Matrix::Create(matrix->rows, 1);
    for (size_t j = 0; j < delta->rows; j++) {
        delta->Set(random.Generate(), j);
    }
    delta->DeviceTransfer(delta->device_alloc, delta);
    matrices.push_back(delta);

    Matrix* multiply_t4 = Matrix::MultiplyGPU(gradient, delta);
    matrices.push_back(multiply_t4);

    multiply_t4->PrintMatrix("Gradient Solution: ");

    return matrices;
}