#include "cuda_interop_helper.cuh"

__host__ cudaError_t CudaInterop::TestMethod() {
	cudaError_t cuda_status = cudaSuccess;

	CrossMemoryHandle test = InteropMemoryHandler::CrossMemoryHandles()[0];

	Grid::SimulateGPU(grid_, cuda_stream_); //WIP
	cuda_status = PostExecutionChecks(cuda_status, "MPMKernel");

	float* host_test = new float();

	cuda_status = cudaMemcpyAsync(host_test, &((Vector3D*)test.cuda_device_ptr)[163].dim[0], sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_);

	s_stream << "Host Test: " << *host_test << std::endl; //WIP, DEBUG!
	ProgramLog::OutputLine(s_stream); //WIP, DEBUG!

	delete host_test;

	return cuda_status;
}

__host__ cudaError_t CudaInterop::BulkInitializationTest(VkSemaphore& wait_semaphore, VkSemaphore& signal_semaphore, const size_t size) {
	cudaError_t cuda_status = cudaSuccess;

	cuda_status = CreateStream();
	ProgramLog::OutputLine("Creating CUDA async stream!\n");

	InteropMemoryHandler::AddMemoryHandle(grid_->GetParticleCount(), sizeof(Vector3D), true);
	ProgramLog::OutputLine("Grid Particle Count: " + std::to_string(grid_->GetParticleCount()));

	CUresult cuda_result = InteropMemoryHandler::CreateNewAllocation();
	grid_->particle_position_device_ = (Vector3D*) InteropMemoryHandler::CrossMemoryHandles()[0].cuda_device_ptr;

	ProgramLog::OutputLine("Setting up simulation interop allocations!");

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