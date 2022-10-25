#include "mpm.cuh"

__global__ void UpdateCell(Grid* grid, Matrix* momentum_matrix, Matrix* cell_dist_matrix, Matrix* momentum) {
	//Particle Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current position

	//Current particle variables
	Vector3D position = grid->GetPosition(incident);
	Vector3D cell_idx = position.Truncate();

	Vector3D cell_difference = (position - cell_idx) - 0.5f;

	Vector3D* weights = GetWeights(cell_difference);

	const size_t traversal_length = 27;

	for (size_t i = 0; i < traversal_length; i++) {
		int x_weight_idx = (i / 9) % 2,
			y_weight_idx = (i / 3) % 2,
			z_weight_idx = i % 2;

		float weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();
		Vector3D cell_position(cell_idx.x() + x_weight_idx - 1,
			cell_idx.y() + y_weight_idx - 1,
			cell_idx.z() + z_weight_idx - 1);

		Vector3D cell_dist = (cell_position - position) + 0.5f;
		for (int j = 0; j < 2; j++) {
			cell_dist_matrix->Get(j) = cell_dist.dim[j];
		}

		Matrix::CopyMatrixOnPointer(momentum_matrix, grid->GetMomentum(incident));

		//Maybe separate this later, and split the UpdateCell into determined chunks?
		//The idea is separating the matrix multiplication (without dynamic parallelism) will significantly increase perf
		// CuBLAS with coalesced memory for strided or parallelized multiplication with tensor cores?
		//
		//Maybe we don't need cell_dist_matrix, and can just use the vector?
		//Maybe also for momentum? Momentum is a 3x1, so Q would not have to be assigned after computation

		for (size_t l = 0; l < cell_dist_matrix->columns; l++) {
			for (size_t j = 0; j < grid->GetMomentum(incident).rows; j++) {
				unsigned int idx = j * momentum->columns + l;
				if (y_bounds < momentum->rows && x_bounds < momentum->columns) {
					float Pvalue = 0.0f;
					for (size_t k = 0; k < momentum_matrix->columns; ++k) {
						Pvalue += momentum_matrix->Get(j * momentum_matrix->columns + k) * cell_dist_matrix->Get(k * cell_dist_matrix->columns + l);
					}
					momentum->Get(idx) = Pvalue;
					//printf("%d\n", idx);
				}
			}
		}

		IndexPair cell_incident(cell_position.x(), cell_position.y(), cell_position.z());

		Vector3D Q = Vector3D(momentum->Get(0), momentum->Get(1), momentum->Get(2));

		float mass_contribution = weight * grid->GetCellMass(cell_incident);

		grid->GetCellMass(cell_incident) += mass_contribution;
		grid->GetCellVelocity(cell_incident) += (grid->GetVelocity(cell_incident) + Q) * mass_contribution;
	}
}