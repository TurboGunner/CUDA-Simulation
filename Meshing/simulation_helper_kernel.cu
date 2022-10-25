#include "mpm.cuh"

//Maybe make weights a shared memory allocation? Maybe a static shared memory context that is accessible through all the sim kernels?
__device__ Vector3D* GetWeights(Vector3D cell_difference) { //Returns weights shared
	Vector3D weights[3]{}; //Array of weights

	weights[0] = (cell_difference.Negative() + 0.5f).Squared() * 0.5f;
	weights[1] = cell_difference.Squared().Negative() + 0.75f;
	weights[2] = (cell_difference + 0.5f).Squared() * 0.5f;

	return weights;
}


__global__ void UpdateGrid(Grid* grid) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current Index

	grid->GetCellVelocity(incident) /= grid->GetCellMass(incident); //Converting momentum to velocity

	//Applying gravity to velocity
	Vector3D gravity_vector(0.0f, 0.0f, grid->gravity);
	grid->GetCellVelocity(incident) += (gravity_vector * grid->dt);

	//Boundary Conditions
	if (x_bounds < 2 || x_bounds > grid->side_size_ - 3) {
		grid->GetCellVelocity(incident).dim[0] = 0.0f;
	}
	if (y_bounds < 2 || y_bounds > grid->side_size_ - 3) {
		grid->GetCellVelocity(incident).dim[1] = 0.0f;
	}
	if (z_bounds < 2 || z_bounds > grid->side_size_ - 3) {
		grid->GetCellVelocity(incident).dim[2] = 0.0f;
	}
}

__global__ void ClearGrid(Grid* grid) {
	//Cell Boundaries
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z_bounds = blockIdx.z * blockDim.z + threadIdx.z;

	IndexPair incident(x_bounds, y_bounds, z_bounds); //Current position

	grid->GetCellMass(incident) = 0;
	grid->GetCellVelocity(incident).Reset();
}