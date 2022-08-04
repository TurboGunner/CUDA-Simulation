#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "rayhit_logic.cuh"
#include "material_data.cuh"
#include "sphere_primitive.cuh"
#include "vector_helpers.cuh"
#include "material.cuh"
#include "camera.cuh"
#include "ray.cuh"

#include "../CUDATest/handler_methods.hpp"

__device__ inline Vector3D Color(const Ray& ray, MaterialData* data, curandState* rand_state) {
	Ray ray_current = ray;
	RayHit hit;
	Vector3D cur_attenuation(1.0f, 1.0f, 1.0f);

	for (int i = 0; i < 50; i++) {
		if (data->Get()->Hit(ray, 0.001f, FLT_MAX, hit)) {
			Ray scatter;
			Vector3D attenuation;
			if (hit.mat_ptr->Scatter(ray_current, hit, attenuation, scatter, rand_state)) {
				cur_attenuation = MultiplyVector(cur_attenuation, attenuation);
				ray_current = scatter;
			}
			else {
				return Vector3D(0.0f, 0.0f, 0.0f);
			}
		}
		else {
			float t = 0.5f * (UnitVector(ray.Direction()).y + 1.0f);
			Vector3D intermediate1 = MultiplyByScalar(Vector3D(1.0f, 1.0f, 1.0f), 1.0f - t);
			Vector3D intermediate2 = MultiplyByScalar(Vector3D(0.5f, 0.7f, 1.0f), t);

			return MultiplyVector(cur_attenuation, AddVector(intermediate1, intermediate2));
		}
	}
	return Vector3D(0.0f, 0.0f, 0.0f);
}

__global__ inline void Render(Vector3D* frame_buffer, uint2 size, int ns, curandState* rand_states, Camera* camera, MaterialData* data) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x_bounds >= size.x) || (y_bounds >= size.y)) {
		return;
	}

	int IX = y_bounds * size.x + x_bounds;
	curandState rand_state = rand_states[IX];

	Vector3D color;

	float u, v;
	Ray ray;

	for (int i = 0; i < ns; i++) {
		u = float(x_bounds + curand_uniform(&rand_state)) / float(size.x);
		v = float(y_bounds + curand_uniform(&rand_state)) / float(size.y);

		ray = camera->GetRay(u, v, &rand_state);
		color = AddVector(color, Color(ray, data, &rand_state));
	}

	frame_buffer[IX] = Color(ray, data, &rand_state);
}

__global__ void AssignRandom(uint2 size, curandState* rand_state) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x_bounds >= size.x) || (y_bounds >= size.y)) {
		return;
	}
	int pixel_index = y_bounds * size.x + x_bounds;

	curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

inline Vector3D* AllocateTexture(uint2 size, cudaError_t& cuda_status) {
	size_t frame_buffer_size = 3 * (size.x * size.y) * sizeof(Vector3D);

	curandState* rand_states;
	cuda_status = cudaMalloc((void**)&rand_states, (size.x * size.y) * sizeof(curandState));

	Vector3D* frame_buffer, * frame_buffer_host;
	cuda_status = cudaMalloc(&frame_buffer, frame_buffer_size);
	cuda_status = cudaMallocHost(&frame_buffer_host, frame_buffer_size);

	Hitable* hitable_list, *world;

	cuda_status = cudaMalloc((void**)&hitable_list, 2 * sizeof(Hitable*));
	cuda_status = cudaMalloc((void**)&world, sizeof(Hitable*));

	MaterialData* data = new MaterialData(2);

	Camera* camera;

	dim3 blocks, threads;
	ThreadAllocator2D(blocks, threads, size.x);
	AssignRandom<<<blocks, threads>>> (size, rand_states);
	Render<<<blocks, threads>>> (frame_buffer, size, 50, rand_states, camera, world);

	cuda_status = PostExecutionChecks(cuda_status, "RenderKernel", true);

	cuda_status = CopyFunction("RenderKernel", frame_buffer_host, frame_buffer, cudaMemcpyDeviceToHost, cuda_status, frame_buffer_size);

	return frame_buffer;
}

__global__ inline void CreateWorld(MaterialData* data, Camera* camera, uint2 size) {

	Material* mat1 = new Lambertian(Vector3D(0.1f, 0.2f, 0.5f));

	Vector3D lookfrom(3.0f, 3.0f, 2.0f);
	Vector3D lookat(0.0f, 0.0f, -1.0f);
	float dist_to_focus = Length(SubtractVector(lookfrom, lookat));
	float aperture = 2.0;

	camera = new Camera(lookfrom, lookat, Vector3D(0.0f, 1.0f, 0.0f), 20.0f, float(size.x) / float(size.y), aperture, dist_to_focus);
}