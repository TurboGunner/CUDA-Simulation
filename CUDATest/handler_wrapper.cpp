#include "handler_wrapper.hpp"

#include <stdexcept>
#include <iostream>

CudaMethodHandler::CudaMethodHandler(const unsigned int& alloc_size, string method_name) {
	if (alloc_size < 1) {
		throw std::invalid_argument("Invalid size argument. Must be 1 or greater.");
	}
	alloc_size_ = alloc_size;
}

void CudaMethodHandler::AllocateCopyPointers() {
	if (float_copy_ptrs_.size() > 0) {
		CudaMemoryAllocator(float_copy_ptrs_, alloc_size_, sizeof(float));
		std::cout << "Allocated " << alloc_size_ * sizeof(float) * float_copy_ptrs_.size() << " bytes of memory!" << std::endl;
	}
	if (float3_copy_ptrs_.size() > 0) {
		CudaMemoryAllocator(float3_copy_ptrs_, alloc_size_, sizeof(float3));
		std::cout << "Allocated " << alloc_size_ * sizeof(float3) * float3_copy_ptrs_.size() << " bytes of memory!" << std::endl;
	}
}

CudaMethodHandler::~CudaMethodHandler() {
	//CudaMemoryFreer(float_copy_ptrs_);
	//CudaMemoryFreer(float3_copy_ptrs_);
}

cudaError_t CudaMethodHandler::CopyToMemory(cudaMemcpyKind mode, cudaError_t status) {
	cudaError_t cuda_status = status;
	for (size_t i = 0; i < float_copy_ptrs_.size() && cuda_status == cudaSuccess; i++) {
		cuda_status = CopyFunction("cudaMemcpy failed at vector position! " + std::to_string(i), float_copy_ptrs_.at(i).get(), float_ptrs_.at(i).get(),
			cudaMemcpyHostToDevice, cuda_status, (size_t)alloc_size_,
			sizeof(float));
	}
	for (size_t i = 0; i < float3_copy_ptrs_.size() && cuda_status == cudaSuccess; i++) {
		cuda_status = CopyFunction("cudaMemcpy failed at vector position! " + std::to_string(i), float3_copy_ptrs_.at(i).get(), float3_ptrs_.at(i).get(),
			cudaMemcpyHostToDevice, cuda_status, (size_t)alloc_size_,
			sizeof(float3));
	}
	return cuda_status;
}

cudaError_t CudaMethodHandler::PostExecutionChecks(cudaError_t status) {
	cudaError_t cuda_status = status;
	if (cuda_status == cudaSuccess) {
		function<cudaError_t()> error_check_func = []() { return cudaGetLastError(); };
		cuda_status = WrapperFunction(error_check_func, "cudaGetLastError (kernel launch)", method_name_, cuda_status);

		function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
		cuda_status = WrapperFunction(sync_func, "cudaDeviceSynchronize", method_name_, cuda_status);
	}
	return cuda_status;
}