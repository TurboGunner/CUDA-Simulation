#include "mpm.cuh"

__global__ void SimulateGrid(Grid* grid, Matrix* stress_matrix, Matrix* momentum, Matrix* viscosity_term) { //Maybe no momentum matrix as well?
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

	int x_weight_idx = 0,
		y_weight_idx = 0,
		z_weight_idx = 0;

	float weight = 0.0f;

	Vector3D cell_position;

	IndexPair cell_incident;

	//Estimation of particle volume by summing up the particle neighborhood weighted mass contribution
	for (size_t i = 0; i < traversal_length; i++) { //NOTE: INCREMENT ORDER
		x_weight_idx = (i / 6) % 2;
		y_weight_idx = (i / 3) % 2;
		z_weight_idx = i % 2;

		weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();

		cell_position = Vector3D(cell_idx.x() + x_weight_idx - 1,
			cell_idx.y() + y_weight_idx - 1,
			cell_idx.z() + z_weight_idx - 1);

		cell_incident = IndexPair(cell_position.x(), cell_position.y(), cell_position.z());

		density += grid->GetCellMass(cell_incident) * weight;
	}

	float volume = grid->GetParticleMass(incident) / density;
	//Max constraints is to ensure particles do not absorb into each other due to negative pressures (divergence sink limiter)
	float pressure = fmax(-0.1f, grid->eos_stiffness * (pow(density / grid->rest_density, grid->eos_power) - 1));

	/* Velocity gradient, or dudv
	* Is specifically where the derivative of the quadratic polynomial is linear
	*/
	Matrix strain_matrix = grid->GetMomentum(incident);

	float trace = strain_matrix.Get(2, 0) + strain_matrix.Get(1, 1) + strain_matrix.Get(0, 2);

	for (size_t i = 0; i < strain_matrix.rows; i++) {
		size_t reverse_idx = strain_matrix.rows - i - 1;
		strain_matrix.Get(i, reverse_idx) = trace;
		stress_matrix->Set(-pressure, i, reverse_idx);
	}

	Matrix::CopyMatrixOnPointer(viscosity_term, strain_matrix);

	Matrix::MultiplyScalarOnPointer(viscosity_term, grid->dynamic_viscosity);

	//Matrix viscosity_term = strain_matrix * grid->dynamic_viscosity;

	Matrix::AddOnPointer(stress_matrix, *viscosity_term);

	Matrix::MultiplyScalarOnPointer(stress_matrix, -volume * 4 * grid->dt); //Final Constitutive Equation for Isotropic Fluid

	for (size_t i = 0; i < traversal_length; i++) { //NOTE: INCREMENT ORDER
		x_weight_idx = (i / 6) % 2;
		y_weight_idx = (i / 3) % 2;
		z_weight_idx = i % 2;

		weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();

		cell_position = Vector3D(cell_idx.x() + x_weight_idx - 1,
			cell_idx.y() + y_weight_idx - 1,
			cell_idx.z() + z_weight_idx - 1);

		cell_incident = IndexPair(cell_position.x(), cell_position.y(), cell_position.z());

		Vector3D cell_dist = (cell_position - position) + 0.5f;

		Vector3D constitutive_vector; //NOTE

		for (size_t j = 0; j < 2; j++) {
			constitutive_vector.dim[j] = stress_matrix->Get(j);
		}

		constitutive_vector = constitutive_vector * cell_dist * weight; //Weights constitutive equation and does it based on the closest cell's distance to the particle

		grid->GetCellVelocity(cell_incident) += constitutive_vector;
	}
}