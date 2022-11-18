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

	Vector3D weights[3] {}; //Array of weights, testing a different method here

	weights[0] = (cell_difference.Negative() + 0.5f).Squared() * 0.5f;
	weights[1] = cell_difference.Squared().Negative() + 0.75f;
	weights[2] = (cell_difference + 0.5f).Squared() * 0.5f;

	const size_t traversal_length = 3;

	float density = 0.0f;

	float weight = 0.0f;

	//Estimation of particle volume by summing up the particle neighborhood weighted mass contribution
	for (int x_weight_idx = 0; x_weight_idx < traversal_length; ++x_weight_idx) {
		for (int y_weight_idx = 0; y_weight_idx < traversal_length; ++y_weight_idx) {
			for (int z_weight_idx = 0; z_weight_idx < traversal_length; ++z_weight_idx) {

				weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();

				Vector3D cell_position = Vector3D(cell_idx.x() + x_weight_idx - 1,
					cell_idx.y() + y_weight_idx - 1,
					cell_idx.z() + z_weight_idx - 1);

				IndexPair cell_incident = IndexPair(cell_position.x(), cell_position.y(), cell_position.z());

				density += grid->GetCellMass(cell_incident) * weight;
			}
		}
	}

	float volume = grid->GetParticleMass(incident) / density;
	//Max constraints is to ensure particles do not absorb into each other due to negative pressures (divergence sink limiter)
	float pressure = fmax(-0.1f, grid->eos_stiffness * (powf(density / grid->rest_density, grid->eos_power) - 1));

	/* Velocity gradient, or dudv
	* Is specifically where the derivative of the quadratic polynomial is linear
	*/
	Matrix<3, 3> stress_matrix = {};
	Matrix<3, 3> strain_matrix = {};

	//Initial assignment to the stress matrix, diagonal matrix
	stress_matrix.Set(-pressure, 0, 0);
	stress_matrix.Set(-pressure, 1, 1);
	stress_matrix.Set(-pressure, 2, 2);

	strain_matrix = grid->GetMomentum(incident);

	float trace = strain_matrix.Get(0, 0) + strain_matrix.Get(1, 1) + strain_matrix.Get(2, 2); //Trace of the strain tensor

	//Diagonal assignments to the strain matrix
	strain_matrix.Set(trace, 0, 0);
	strain_matrix.Set(trace, 1, 1);
	strain_matrix.Set(trace, 2, 2);

	strain_matrix *= grid->dynamic_viscosity;
	stress_matrix += strain_matrix;

	stress_matrix *= (-volume * 4.0f * grid->dt);

	Vector3D constitutive_vector = {}; //Final Constitutive Equation for Isotropic Fluid

	for (int x_weight_idx = 0; x_weight_idx < traversal_length; ++x_weight_idx) {
		for (int y_weight_idx = 0; y_weight_idx < traversal_length; ++y_weight_idx) {
			for (int z_weight_idx = 0; z_weight_idx < traversal_length; ++z_weight_idx) {

				weight = weights[x_weight_idx].x() * weights[y_weight_idx].y() * weights[z_weight_idx].z();

				Vector3D cell_position = Vector3D(cell_idx.x() + x_weight_idx - 1,
					cell_idx.y() + y_weight_idx - 1,
					cell_idx.z() + z_weight_idx - 1);

				IndexPair cell_incident = IndexPair(cell_position.x(), cell_position.y(), cell_position.z());

				Matrix<3, 3> weighted_stress_matrix = stress_matrix.UnrolledFixedMultiplyScalar(weight);

				Vector3D cell_dist = (cell_position - position) + 0.5f;

				constitutive_vector = UnrolledFixedMV(weighted_stress_matrix, cell_dist);

				grid->GetCellVelocity(cell_incident) += constitutive_vector;
			}
		}
	}
}