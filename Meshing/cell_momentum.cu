#include "mpm.cuh"

__global__ void UpdateCell(Grid* grid, Matrix* momentum_matrix, Matrix* cell_dist_matrix, Matrix* momentum) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current position

	//Current particle variables
	for (size_t k = 0; k < grid->GetResolution(); k++) {
		Particle* particle = grid->GetParticle(incident, k);
		Vector3D position = particle->position;
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
			for (int j = 0; j < cell_dist_matrix->columns; j++) {
				cell_dist_matrix->Get(j) = cell_dist.dim[j];
			}

			dim3 blocks, threads;

			threads = dim3(1, 1);
			blocks = dim3(ceil((1.0 * cell_dist_matrix->columns)), ceil((1.0f * particle->momentum.rows)), 1); //NOTE: May be backwards

			*momentum_matrix = particle->momentum;

			MultiplyKernel<<<blocks, threads >>> (momentum_matrix, cell_dist_matrix, momentum);

			Vector3D Q = Vector3D(momentum->Get(0), momentum->Get(1), momentum->Get(2));

			float mass_contribution = weight * particle->mass;

			Cell* cell = grid->GetCell(incident);

			cell->mass += mass_contribution;
			cell->velocity = cell->velocity + (particle->velocity + Q) * mass_contribution;

			//cell_dist_matrix->Destroy();
			//momentum->Destroy();
			//momentum_matrix->Destroy();
		}
	}
}