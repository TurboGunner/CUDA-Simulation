#include "mpm.cuh"

__host__ cudaError_t InitializeGridHost(Grid* grid) { //NOTE
	
	return grid->DeviceTransfer(grid);
}