#include "mpm.cuh"

__global__ void SimulateGrid(Grid* grid) { //Maybe no momentum matrix as well?
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

	float density = 0.0f;

	int x_weight_idx = 0,
		y_weight_idx = 0,
		z_weight_idx = 0;

	float weight = 0.0f;

	Vector3D cell_position = {};

	IndexPair cell_incident = {};

	Vector3D cell_dist = {};

	int i = 0, j = 0;

	//Estimation of particle volume by summing up the particle neighborhood weighted mass contribution
	for (x_weight_idx = 0; x_weight_idx < traversal_length; ++x_weight_idx) {
		for (y_weight_idx = 0; y_weight_idx < traversal_length; ++y_weight_idx) {
			for (z_weight_idx = 0; z_weight_idx < traversal_length; ++z_weight_idx) {

				weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();

				cell_position = Vector3D(cell_idx.x() + x_weight_idx - 1,
					cell_idx.y() + y_weight_idx - 1,
					cell_idx.z() + z_weight_idx - 1);

				cell_incident = IndexPair(cell_position.x(), cell_position.y(), cell_position.z());

				density += grid->GetCellMass(cell_incident) * weight;
			}
		}
	}

	float volume = grid->GetParticleMass(incident) / density;
	//Max constraints is to ensure particles do not absorb into each other due to negative pressures (divergence sink limiter)
	float pressure = fmax(-0.1f, grid->eos_stiffness * (pow(density / grid->rest_density, grid->eos_power) - 1));

	/* Velocity gradient, or dudv
	* Is specifically where the derivative of the quadratic polynomial is linear
	*/
	Matrix stress_matrix(3, 3, true, false);
	Matrix strain_matrix(3, 3, true, false);

	strain_matrix = grid->GetMomentum(incident);

	//Matrix::CopyMatrixOnPointer(strain_matrix, grid->GetMomentum(incident));

	float trace = strain_matrix.Get(2, 0) + strain_matrix.Get(1, 1) + strain_matrix.Get(0, 2);

	size_t reverse_idx = 0;
	for (size_t i = 0; i < strain_matrix.rows; i++) {
		reverse_idx = strain_matrix.rows - i - 1;
		strain_matrix.Get(i, reverse_idx) = trace;
		stress_matrix.Set(-pressure, i, reverse_idx);
	}

	strain_matrix *= grid->dynamic_viscosity;
	stress_matrix += strain_matrix;
	//Matrix::AddOnPointer(stress_matrix, *strain_matrix);

	//Matrix::MultiplyScalarOnPointer(stress_matrix, -volume * 4 * grid->dt); //Final Constitutive Equation for Isotropic Fluid
	stress_matrix *= (-volume * 4.0f * grid->dt);

	Vector3D constitutive_vector = {};

	for (x_weight_idx = 0; x_weight_idx < traversal_length; ++x_weight_idx) {
		for (y_weight_idx = 0; y_weight_idx < traversal_length; ++y_weight_idx) {
			for (z_weight_idx = 0; z_weight_idx < traversal_length; ++z_weight_idx) {

				weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();

				cell_position = Vector3D(cell_idx.x() + x_weight_idx - 1,
					cell_idx.y() + y_weight_idx - 1,
					cell_idx.z() + z_weight_idx - 1);

				cell_incident = IndexPair(cell_position.x(), cell_position.y(), cell_position.z());

				cell_dist = (cell_position - position) + 0.5f;

				for (i = 0; i < grid->GetMomentum(incident).rows; ++i) {
					//if (y_bounds < 3 && x_bounds < 1) {
						float p_value = 0.0f;
						for (j = 0; j < stress_matrix.columns; ++j) {
							p_value += (stress_matrix.Get(i * stress_matrix.columns + j) * weight) * cell_dist.dim[j];
						}
						constitutive_vector.dim[i] = p_value; //Number of columns is always 1, so this can be disregarded, idx can just be i
					//}
				}

				grid->GetCellVelocity(cell_incident) += constitutive_vector;
			}
		}
	}
}