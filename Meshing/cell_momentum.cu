#include "mpm.cuh"

__global__ void UpdateCell(Grid* grid) {
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

	const size_t traversal_length = 3;

	int x_weight_idx = 0,
		y_weight_idx = 0,
		z_weight_idx = 0;

	int i = 0, j = 0;

	Vector3D Q = {};

	for (x_weight_idx = 0; x_weight_idx < traversal_length; ++x_weight_idx) {
		for (y_weight_idx = 0; y_weight_idx < traversal_length; ++y_weight_idx) {
			for (z_weight_idx = 0; z_weight_idx < traversal_length; ++z_weight_idx) {

				float weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();

				Vector3D cell_position(cell_idx.x() + x_weight_idx - 1,
					cell_idx.y() + y_weight_idx - 1,
					cell_idx.z() + z_weight_idx - 1);

				IndexPair cell_incident(cell_idx.x() + x_weight_idx - 1, cell_idx.y() + y_weight_idx - 1, cell_idx.z() + z_weight_idx - 1);

				Vector3D cell_dist = (cell_position - position) + 0.5f;

				//Matrix::CopyMatrixOnPointer(momentum_matrix, grid->GetMomentum(incident));

				Matrix& momentum_matrix = grid->GetMomentum(incident);
				printf("%d\n", grid->GetMomentum(incident).rows);

				//Maybe separate this later, and split the UpdateCell into determined chunks?
				//The idea is separating the matrix multiplication (without dynamic parallelism) will significantly increase perf
				// CuBLAS with coalesced memory for strided or parallelized multiplication with tensor cores?

					for (i = 0; j < grid->GetMomentum(incident).rows; ++i) {
						//unsigned int idx = i * momentum->columns;
						//if (y_bounds < momentum->rows && x_bounds < momentum->columns) {
							float result = 0.0f;
							for (j = 0; j < momentum_matrix.columns; ++j) {
								//result += momentum_matrix.Get(i * momentum_matrix.columns + j) * cell_dist.dim[j];
							}
							Q.dim[i] = result; //See Gradient Solve MM portion for rationale
						//}
					}

				float mass_contribution = weight * grid->GetParticleMass(incident); //Fixed, this was grid mass

				//Updates mass and velocity
				grid->GetCellMass(cell_incident) += mass_contribution;
				grid->GetCellVelocity(cell_incident) += (grid->GetVelocity(incident) + Q) * mass_contribution;
			}
		}
	}
}