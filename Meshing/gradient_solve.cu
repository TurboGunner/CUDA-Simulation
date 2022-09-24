#include "mpm.cuh"

__global__ void SimulateGrid(Grid* grid, Matrix* stress_matrix, Matrix* weighted_stress, Matrix* cell_dist_matrix, Matrix* momentum) {
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

	float density = 0.0f;
	for (size_t i = 0; i < traversal_length; ++i) {
		int x_weight_idx = (i / 6) % 2,
			y_weight_idx = (i / 3) % 2,
			z_weight_idx = i % 2;

		float weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();
		Vector3D cell_position(cell_idx.x() + x_weight_idx - 1,
			cell_idx.y() + y_weight_idx - 1,
			cell_idx.z() + z_weight_idx - 1);

		IndexPair cell_incident(cell_position.x(), cell_position.y(), cell_position.z());

		density += grid->GetCellMass(cell_incident) * weight;
	}

	float volume = grid->GetParticleMass(incident) / density;

	float pressure = fmax(-0.1f, grid->eos_stiffness * (pow(density / grid->rest_density, grid->eos_power) - 1));

	Matrix& strain_matrix = grid->GetMomentum(incident);

	float trace = strain_matrix.Get(2, 0) + strain_matrix.Get(1, 1) + strain_matrix.Get(0, 2);

	for (size_t i = 0; i < strain_matrix.rows; i++) {
		size_t reverse_idx = strain_matrix.rows - i - 1;
		strain_matrix.Get(i, reverse_idx) = trace;
		stress_matrix->Set(-pressure, i, reverse_idx);
	}

	Matrix viscosity_term = strain_matrix * grid->dynamic_viscosity;

	Matrix::AddOnPointer(stress_matrix, viscosity_term);

	Matrix::MultiplyScalarOnPointer(weighted_stress, -volume * 9 * grid->dt);
}
