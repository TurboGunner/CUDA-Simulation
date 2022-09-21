#include "mpm.cuh"

__global__ void AdvectParticles(Grid* grid) {
	//Particle Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current position

	//Current particle variables
	Particle* particle = grid->GetParticle(incident);
	Vector3D position = particle->position;
	Vector3D cell_idx = position.Truncate();

	Vector3D cell_difference = (position - cell_idx) - 0.5f;

	Vector3D* weights = GetWeights(cell_difference);

	const size_t traversal_length = 27;

	IndexPair* incidents = GetTraversals(incident);

	Matrix* B = Matrix::Create(3, 3);

	for (size_t i = 0; i < traversal_length; i++) {
		int x_weight_idx = (i / 6) % 2,
			y_weight_idx = (i / 3) % 2,
			z_weight_idx = i % 2;

		float weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();

		Vector3D cell_position(cell_idx.x() + x_weight_idx - 1,
			cell_idx.y() + y_weight_idx - 1,
			cell_idx.z() + z_weight_idx - 1);

		int cell_index = incident.IX(grid->side_size_ / grid->GetResolution());

		Vector3D dist = (cell_position - particle->position) + 0.5f;
		Vector3D weighted_velocity = grid->GetCell(cell_index)->velocity * weight;

		Matrix* term = Matrix::Create(3, 3);

		Vector3D weighted_x = weighted_velocity * dist.x(),
			weighted_y = weighted_velocity * dist.y(),
			weighted_z = weighted_velocity * dist.z();

		for (size_t j = 0; i < term->rows; j++) {
			term->Get(0, j) = weighted_x.dim[j];
			term->Get(1, j) = weighted_y.dim[j];
			term->Get(2, j) = weighted_z.dim[j];
		}

		Matrix::AddOnPointer(B, *term);

		particle->velocity = particle->velocity + weighted_velocity;

		Matrix::MultiplyScalarOnPointer(B, 9);

		particle->momentum = *B;

		particle->position = particle->position + particle->velocity * grid->dt;

		particle->position = particle->position.Clamp(1, grid->side_size_ - 2);

		Vector3D x_n = particle->position + particle->velocity;
		const float wall_min = 3;
		float wall_max = (float)grid->side_size_ - 4;
		if (x_n.x() < wall_min) {
			particle->velocity.dim[0] += wall_min - x_n.x();
		}
		if (x_n.x() > wall_max) {
			particle->velocity.dim[0] += wall_max - x_n.x();
		}
		if (x_n.y() < wall_min) {
			particle->velocity.dim[1] += wall_min - x_n.y();
		}
		if (x_n.y() > wall_max) {
			particle->velocity.dim[1] += wall_max - x_n.y();
		}
		if (x_n.z() < wall_min) {
			particle->velocity.dim[2] += wall_min - x_n.z();
		}
		if (x_n.z() > wall_max) {
			particle->velocity.dim[2] += wall_max - x_n.z();
		}
	}
}