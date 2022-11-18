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

	Vector3D weights[3] {}; //Array of weights, testing a different method here

	weights[0] = (cell_difference.Negative() + 0.5f).Squared() * 0.5f;
	weights[1] = cell_difference.Squared().Negative() + 0.75f;
	weights[2] = (cell_difference + 0.5f).Squared() * 0.5f;

	const size_t traversal_length = 3;

	Vector3D Q = {};

	for (int x_weight_idx = 0; x_weight_idx < traversal_length; ++x_weight_idx) {
		for (int y_weight_idx = 0; y_weight_idx < traversal_length; ++y_weight_idx) {
			for (int z_weight_idx = 0; z_weight_idx < traversal_length; ++z_weight_idx) {

				float weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();

				Vector3D cell_position(cell_idx.x() + x_weight_idx - 1,
					cell_idx.y() + y_weight_idx - 1,
					cell_idx.z() + z_weight_idx - 1);

				IndexPair cell_incident(cell_idx.x() + x_weight_idx - 1, cell_idx.y() + y_weight_idx - 1, cell_idx.z() + z_weight_idx - 1);

				Vector3D cell_dist = (cell_position - position) + 0.5f;

				Matrix<3, 3>& momentum_matrix = grid->GetMomentum(incident);

				Q = UnrolledFixedMV(momentum_matrix, cell_dist);

				float mass_contribution = weight * grid->GetParticleMass(incident); //Fixed, this was grid mass

				//Updates mass and velocity
				grid->GetCellMass(cell_incident) += mass_contribution;
				grid->GetCellVelocity(cell_incident) += (grid->GetVelocity(incident) + Q) * mass_contribution;
			}
		}
	}
}