#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "rayhit_logic.cuh"
#include "sphere_primitive.cuh"
#include "vector_helpers.cuh"
#include "material.cuh"
#include "camera.cuh"

#include "handler_methods.hpp"

__global__ inline void Render(float3* frame_buffer, uint2 size, int ns, curandState* rand_states, Camera** camera, Hitable** world) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x_bounds >= size.x) || (y_bounds >= size.y)) {
		return;
	}
	int IX = y_bounds * size.x + x_bounds;
	curandState rand_state = rand_state[IX];

	float3 color;

	float u, v;
	Ray r;

	for (size_t i = 0; i < ns; i++) {
		u = float(x_bounds + curand_uniform(&rand_state)) / float(size.x);
		v = float(y_bounds + curand_uniform(&rand_state)) / float(size.y);

		r = (*camera)->GetRay(u, v);
	}
	
	frame_buffer[IX] = Color(r, world);
}

__global__ void AssignRandom(uint2 size, curandState* rand_state) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x_bounds >= size.x) || (y_bounds >= size.y)) return;
	int pixel_index = y_bounds * size.x + x_bounds;

	curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void FreeWorld(Hitable** hitable_list, Hitable** world) {
	delete* (hitable_list);
	delete* (hitable_list + 1);
	delete* world;
}

inline float3* AllocateTexture(uint2 size, cudaError_t& cuda_status) {
	size_t frame_buffer_size = 3 * (size.x * size.y) * sizeof(float3);

	curandState* rand_state;
	cuda_status = cudaMalloc((void**)&rand_state, (size.x * size.y) * sizeof(curandState));

	float3* frame_buffer, *frame_buffer_host;
	cuda_status = cudaMalloc(&frame_buffer, frame_buffer_size);
	cuda_status = cudaMallocHost(&frame_buffer_host, frame_buffer_size);

	Hitable** hitable_list, **world;

	cuda_status = cudaMalloc((void**)&hitable_list, 2 * sizeof(Hitable*));
	cuda_status = cudaMalloc((void**)&world, sizeof(Hitable*));

	dim3 blocks, threads;
	ThreadAllocator2D(blocks, threads, size.x);
	Render<<<blocks, threads>>> (frame_buffer, size, float3(-2.0f, -1.0f, -1.0f),
		float3(4.0f, 0.0f, 0.0f),
		float3(0.0f, 2.0f, 0.0f),
		float3(0.0f, 0.0f, 0.0f),
		world);

	cuda_status = PostExecutionChecks(cuda_status, "RenderKernel", true);

	cuda_status = CopyFunction("RenderKernel", frame_buffer_host, frame_buffer, cudaMemcpyDeviceToHost, cuda_status, frame_buffer_size);

	return frame_buffer;
}

__device__ inline float3 Color(const Ray& ray, Hitable** world) {
	Ray ray_current = ray;
	RayHit hit;

	float attenuation = 1.0f;
	
	if ((*world)->Hit(ray, 0.001f, FLT_MAX, hit)) {
		return MultiplyByScalar(AddByScalar(hit.normal, 1.0f), 0.5f);
	}

	float t = 0.5f * (UnitVector(ray.Direction()).y + 1.0f);
	float3 intermediate1 = MultiplyByScalar(float3(1.0f, 1.0f, 1.0f), 1.0f - t);
	float3 intermediate2 = MultiplyByScalar(float3(0.5f, 0.7f, 1.0f), t);

	return AddVector(intermediate1, intermediate2);
}

__global__ inline void CreateWorld(Hitable** hitable_list, Hitable** world) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*(hitable_list) = new Sphere(float3(0.0f, 0.0f, -1.0f), 0.5f);
		*(hitable_list + 1) = new Sphere(float3(0.0f, -100.5f, -1.0f), 100.0f);
		*world = new HitableList(hitable_list, 2);
	}
}