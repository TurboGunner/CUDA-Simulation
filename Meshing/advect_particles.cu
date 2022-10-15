#include "mpm.cuh"

__global__ void AdvectParticles(Grid* grid, Matrix* B_term, Matrix* weighted_term) {
	//Particle Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current positions

	//NOTE
	grid->GetVelocity(incident).Reset(); //Sets velocity to zero

	//Current particle variables
	Vector3D position = grid->GetPosition(incident);
	Vector3D cell_idx = position.Truncate();

	Vector3D cell_difference = (position - cell_idx) - 0.5f;

	Vector3D* weights = GetWeights(cell_difference);

	const size_t traversal_length = 27;

	for (size_t i = 0; i < traversal_length; i++) {
		int x_weight_idx = (i / 6) % 2,
			y_weight_idx = (i / 3) % 2,
			z_weight_idx = i % 2;

		float weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();

		Vector3D cell_position(cell_idx.x() + x_weight_idx - 1,
			cell_idx.y() + y_weight_idx - 1,
			cell_idx.z() + z_weight_idx - 1);

		IndexPair cell_incident(cell_position.x(), cell_position.y(), cell_position.z());

		Vector3D dist = (cell_position - position) + 0.5f;
		Vector3D weighted_velocity = grid->GetCellVelocity(cell_incident) * weight;

		Vector3D weighted_x = weighted_velocity * dist.x(),
			weighted_y = weighted_velocity * dist.y(),
			weighted_z = weighted_velocity * dist.z();

		for (size_t j = 0; j < 2; j++) { //Full array assignment to weighting terms
			weighted_term->Get(0, j) = weighted_x.dim[j];
			weighted_term->Get(1, j) = weighted_y.dim[j];
			weighted_term->Get(2, j) = weighted_z.dim[j];
		}

		Matrix::AddOnPointer(B_term, *weighted_term);

		grid->GetVelocity(incident) += weighted_velocity;

		Matrix::MultiplyScalarOnPointer(B_term, 9);

		grid->GetMomentum(incident) = *B_term;

		grid->GetPosition(incident) += grid->GetVelocity(incident) * grid->dt;

		grid->GetPosition(incident) = grid->GetPosition(incident).Clamp(1, grid->side_size_ - 2);

		Vector3D position_normalized = grid->GetPosition(incident) + grid->GetVelocity(incident);
		const float wall_min = 3;
		float wall_max = (float) grid->side_size_ - 4;

		MPMBoundaryConditions(grid, incident, position_normalized, wall_min, wall_max);
		//printf("%f\n", grid->GetPosition(incident).dim[0]);
	}
}

__device__ void MPMBoundaryConditions(Grid* grid, IndexPair incident, const Vector3D& position_normalized, const float& wall_min, const float& wall_max) {
	if (position_normalized.x() < wall_min) {
		grid->GetVelocity(incident).dim[0] += wall_min - position_normalized.x();
	}
	if (position_normalized.x() > wall_max) {
		grid->GetVelocity(incident).dim[0] += wall_max - position_normalized.x();
	}
	if (position_normalized.y() < wall_min) {
		grid->GetVelocity(incident).dim[1] += wall_min - position_normalized.y();
	}
	if (position_normalized.y() > wall_max) {
		grid->GetVelocity(incident).dim[1] += wall_max - position_normalized.y();
	}
	if (position_normalized.z() < wall_min) {
		grid->GetVelocity(incident).dim[2] += wall_min - position_normalized.z();
	}
	if (position_normalized.z() > wall_max) {
		grid->GetVelocity(incident).dim[2] += wall_max - position_normalized.z();
	}
}