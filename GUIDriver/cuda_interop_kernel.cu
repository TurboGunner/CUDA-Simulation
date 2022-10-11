#include "cuda_interop_helper.cuh"

__global__ void TestKernel(float* data) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	data[x_bounds] = x_bounds;
}

__host__ cudaError_t CudaInterop::TestMethod(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore) {
	cudaError_t cuda_status = cudaSuccess;

	void*& device_ptr = cross_memory_handles_[0].cuda_device_ptr,
		*& host_ptr = cross_memory_handles_[0].cuda_host_ptr;
	cuda_status = cudaMemsetAsync(device_ptr, 0, cross_memory_handles_[0].TotalAllocationSize(), cuda_stream_);
	CudaExceptionHandler(cuda_status, "CUDAMemsetAsync");

	//dim3 blocks, threads;

	//TestKernel<<<blocks, threads, 0, cuda_stream_>>> ((float*) device_ptr);
	Grid::SimulateGPU(grid_, cuda_stream_); //WIP

	cuda_status = PostExecutionChecks(cuda_status, "MPMKernel");

	//cuda_status = cudaMemcpyAsync(host_ptr, device_ptr, cross_memory_handles_[0].TotalAllocationSize(), cudaMemcpyDeviceToHost, cuda_stream_); //NOTE: TEST

	return cuda_status;
}

__host__ cudaError_t CudaInterop::BulkInitializationTest(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore, const size_t& size) {
	cudaError_t cuda_status = cudaSuccess;

	AddMemoryHandle(size, sizeof(Vector3D)); //Adds memory handle struct

	cuda_status = CreateStream();
	ProgramLog::OutputLine("Creating CUDA async stream!");

	CUresult cuda_result = SimulationSetupAllocations(); //Setups the allocation for the simulation
	ProgramLog::OutputLine("Setting up simulation interop allocations!");

	//for (auto& cross_memory_handle : cross_memory_handles_) {
		//cuda_status = cross_memory_handle.AllocateCudaMemory(); //Allocates CUDA memory across handle structs
	//}
	ProgramLog::OutputLine("CUDA memory allocated successfully!");

	cross_memory_handles_[0].cuda_device_ptr = grid_->particle_velocity_device_;

	cuda_status = InitializeCudaInterop(wait_semaphore, signal_semaphore);

	return cuda_status;
}

__host__ cudaError_t CudaInterop::InteropDrawFrame(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore) {
	cudaExternalSemaphoreWaitParams wait_params = {};
	wait_params.flags = 0;
	wait_params.params.fence.value = 0;

	cudaExternalSemaphoreSignalParams signal_params = {};
	signal_params.flags = 0;
	signal_params.params.fence.value = 0;

	cudaError_t cuda_status = cudaWaitExternalSemaphoresAsync(&cuda_wait_semaphore_, &wait_params, 1, cuda_stream_);
	CudaExceptionHandler(cuda_status, "CUDAWaitExternalSemaphoreAsync");

	cuda_status = TestMethod(wait_semaphore, signal_semaphore);
	CudaExceptionHandler(cuda_status, "ExecuteMethod");

	cuda_status = cudaSignalExternalSemaphoresAsync(&cuda_signal_semaphore_, &signal_params, 1, cuda_stream_);
	CudaExceptionHandler(cuda_status, "CUDASignalExternalSemaphoreAsync");

	return cuda_status;
}

__host__ void CudaInterop::DriverLog(CUresult& cuda_result, const string& label) {
	const char* name_output, *str_output;

	cuda_result = cuGetErrorName(cuda_result, &name_output);
	cuda_result = cuGetErrorString(cuda_result, &str_output);

	s_stream << "CUDA Driver API Error Status for " << label << ": " << name_output << " | CUDA Driver Error String: " << str_output << std::endl;
	ProgramLog::OutputLine(s_stream);
}