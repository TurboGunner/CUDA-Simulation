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

#include <vector>

using std::vector;

static bool run_init = true;

__device__ Vector3D Color(const Ray& ray, MaterialData* data, curandState* rand_state) {
	Ray ray_current = ray;
	Vector3D cur_attenuation(1.0f, 1.0f, 1.0f);

	for (int i = 0; i < 50; i++) {
		RayHit hit;
		if ((*data->Get())->Hit(ray_current, 0.001f, FLT_MAX, hit)) {
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
			float t = 0.5f * (UnitVector(ray_current.Direction()).y() + 1.0f);
			Vector3D intermediate1 = MultiplyByScalar(Vector3D(1.0f, 1.0f, 1.0f), 1.0f - t);
			Vector3D intermediate2 = MultiplyByScalar(Vector3D(0.3f, 0.5f, 1.0f), t);

			return MultiplyVector(cur_attenuation, AddVector(intermediate1, intermediate2));
		}
	}
	return Vector3D(0.0f, 0.0f, 0.0f);
}

__global__ void Render(Vector3D* frame_buffer, uint2 size, int ns, curandState* rand_states, Camera** camera, MaterialData* data) {
	unsigned int x_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_bounds = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x_bounds >= size.x) || (y_bounds >= size.y)) {
		return;
	}

	int IX = y_bounds * size.x + x_bounds;
	curandState rand_state = rand_states[IX];

	Vector3D color(0.0f, 0.0f, 0.0f);

	float u, v;
	Ray ray;

	for (int i = 0; i < ns; i++) {
		u = (float(x_bounds) + curand_uniform(&rand_state)) / float(size.x);
		v = (float(y_bounds) + curand_uniform(&rand_state)) / float(size.y);

		ray = (*camera)->GetRay(u, v, &rand_state);
		color = AddVector(color, Color(ray, data, &rand_state));
	}
	rand_states[IX] = rand_state;

	color = DivideByScalar(color, float(ns));
	color.dim[0] = sqrt(color.dim[0]);
	color.dim[1] = sqrt(color.dim[1]);
	color.dim[2] = sqrt(color.dim[2]);

	frame_buffer[IX] = color;
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

__global__ void CreateWorld(MaterialData* data, Camera** camera, uint2 size) {
	Material* mat1 = new Lambertian(Vector3D(0.1f, 0.5f, 0.5f));
	Sphere* sphere = new Sphere(Vector3D(0.01f, 0.01f, -3.0f), 3.0f, mat1);

	data->Put(0, sphere);

	Vector3D lookfrom(3.0f, 3.0f, 5.0f);
	Vector3D lookat(0.01f, 0.01f, -3.0f);
	float dist_to_focus = Length(SubtractVector(lookfrom, lookat));
	float aperture = 2.0f;

	*camera = new Camera(lookfrom, lookat, Vector3D(0.01f, 1.0f, 0.01f), 20.0f, float(size.x) / float(size.y), aperture, dist_to_focus);
}

Vector3D* AllocateTexture(uint2 size, cudaError_t& cuda_status) {
	size_t frame_buffer_size = 3 * (size.x * size.y) * sizeof(Vector3D);

	curandState* rand_states;
	cuda_status = cudaMalloc((void**)&rand_states, (size.x * size.y) * sizeof(curandState));

	Vector3D* frame_buffer, *frame_buffer_host;

	cuda_status = cudaMalloc(&frame_buffer, frame_buffer_size);
	cuda_status = cudaMallocHost(&frame_buffer_host, frame_buffer_size);

	MaterialData* data = new MaterialData(1);
	MaterialData* data_device = nullptr;

	data->DeviceTransfer(cuda_status, data, data_device);

	Camera** camera;

	cuda_status = cudaMalloc((void**)&camera, sizeof(Camera*));

	dim3 blocks, threads;
	ThreadAllocator2D(blocks, threads, size.x);
	AssignRandom<<<blocks, threads>>> (size, rand_states);

	cudaDeviceSynchronize();

	CreateWorld<<<1, 1>>> (data->device_alloc_, camera, size);
	
	cuda_status = PostExecutionChecks(cuda_status, "InitRenderKernel", true);

	cudaDeviceSynchronize();

	Render<<<blocks, threads>>> (frame_buffer, size, 128, rand_states, camera, data->device_alloc_);

	cuda_status = PostExecutionChecks(cuda_status, "RenderKernel", true);

	cuda_status = CopyFunction("RenderCopyKernel", frame_buffer_host, frame_buffer, cudaMemcpyDeviceToHost, cuda_status, frame_buffer_size);

	cuda_status = cudaFree(frame_buffer);
	cuda_status = cudaFree(camera);
	cuda_status = cudaFree(rand_states);

	//delete data;

	return frame_buffer_host;
}

inline vector<Vector3D> OutputImage(Vector3D* input, uint2 size) {
	vector<Vector3D> output;
	for (size_t i = 0; i < size.x * size.y; i++) {
		output.push_back(input[i]);
	}
	free(input);
}