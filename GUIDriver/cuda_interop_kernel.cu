#include "cuda_interop_helper.cuh"

__host__ cudaError_t CudaInterop::TestMethod() {
	cudaError_t cuda_status = cudaSuccess;

	cuda_status = cudaMemsetAsync(cross_memory_handles_[0].cuda_device_ptr, 0, cross_memory_handles_[0].TotalAllocationSize(), cuda_stream_);
	CudaExceptionHandler(cuda_status, "CUDAMemsetAsync");

	//dim3 blocks, threads;

	Grid::SimulateGPU(grid_, cuda_stream_); //WIP

	cuda_status = PostExecutionChecks(cuda_status, "MPMKernel");

	//cudaStreamSynchronize(cuda_stream_);

	float* host_test = new float();

	cuda_status = cudaMemcpyAsync(host_test, &((Vector3D*) cross_memory_handles_[0].cuda_device_ptr)[163].dim[0], sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_);

	//cudaFree(cross_memory_handles_[0].cuda_device_ptr);

	s_stream << "Host Test: " << *host_test << std::endl; //WIP, DEBUG!
	ProgramLog::OutputLine(s_stream); //WIP, DEBUG!

	s_stream << "Pointer Test: " << cross_memory_handles_[0].vulkan_ptr << std::endl; //WIP, DEBUG!
	ProgramLog::OutputLine(s_stream); //WIP, DEBUG!

	delete host_test;
	//cuda_status = cudaMemcpyAsync(host_ptr, device_ptr, cross_memory_handles_[0].TotalAllocationSize(), cudaMemcpyDeviceToHost, cuda_stream_); //NOTE: TEST

	return cuda_status;
}

__host__ cudaError_t CudaInterop::BulkInitializationTest(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore, const size_t& size) {
	cudaError_t cuda_status = cudaSuccess;

	AddMemoryHandle(size, sizeof(Vector3D)); //Adds memory handle struct

	cuda_status = CreateStream();
	ProgramLog::OutputLine("Creating CUDA async stream!\n");

	cross_memory_handles_[0].cuda_device_ptr = grid_->particle_position_device_;

	CUresult cuda_result = SimulationSetupAllocations(); //Setups the allocation for the simulation
	ProgramLog::OutputLine("Setting up simulation interop allocations!");

	//for (auto& cross_memory_handle : cross_memory_handles_) {
		//cuda_status = cross_memory_handle.AllocateCudaMemory(); //Allocates CUDA memory across handle structs
	//}
	ProgramLog::OutputLine("CUDA memory allocated successfully!");

	cuda_status = InitializeCudaInterop(wait_semaphore, signal_semaphore);

	CudaExceptionHandler(cuda_status, "CUDA Interop");

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

	cuda_status = TestMethod();
	CudaExceptionHandler(cuda_status, "ExecuteMethod");

	cuda_status = cudaSignalExternalSemaphoresAsync(&cuda_signal_semaphore_, &signal_params, 1, cuda_stream_);
	CudaExceptionHandler(cuda_status, "CUDASignalExternalSemaphoreAsync");

	return cuda_status;
}