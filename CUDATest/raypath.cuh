#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "handler_methods.hpp"

__global__ inline void Render(float3* frame_buffer, uint2 size, float3 lower_left_corner, float3 horizontal, float3 vertical, float3 origin, Hitable** world) {
	unsigned int y_bounds = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int x_bounds = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x_bounds >= size.x) || (y_bounds >= size.y)) {
		return;
	}
	int pixel_index = y_bounds * size.x + x_bounds; //Note!
	float u = float(y_bounds) / float(size.x);
	float v = float(x_bounds) / float(size.y);

	float3 intermediate1 = AddByScalar(lower_left_corner, u);
	intermediate1 = MultiplyVector(intermediate1, horizontal);

	float3 intermediate2 = MultiplyByScalar(vertical, v);

	Ray r(origin, AddVector(intermediate1, intermediate2));
	frame_buffer[pixel_index] = Color(r, world);
}

inline cudaError_t AllocateTexture(uint2 size) {
	cudaError_t cuda_status = cudaSuccess;

	size_t frame_buffer_size = 3 * (size.x * size.y) * sizeof(float3);

	float3* frame_buffer, *frame_buffer_host;
	cuda_status = cudaMalloc(&frame_buffer, frame_buffer_size);
	cuda_status = cudaMallocHost(&frame_buffer_host, frame_buffer_size);

	dim3 blocks, threads;
	ThreadAllocator2D(blocks, threads, size.x);
	Render<<<blocks, threads>>> (frame_buffer, size);

	cuda_status = PostExecutionChecks(cuda_status, "RenderKernel", true);

	cuda_status = CopyFunction("RenderKernel", frame_buffer_host, frame_buffer, cudaMemcpyDeviceToHost, cuda_status, frame_buffer_size);

	return cuda_status;
}

struct Ray {
	__device__ Ray() = default;
	__device__ Ray(const float3& start_in, const float3& end_in) {
		start = start_in;
		end = end_in;
	}

	__device__ float3 Origin() const {
		return start;
	}

	__device__ float3 Direction() const {
		return end;
	}

	__device__ float3 PointTowards(const float& t) const {
		float3 multiplier = MultiplyByScalar(end, t);

		multiplier.x += start.x;
		multiplier.y += start.y;
		multiplier.z += start.z;

		return multiplier;
	}

	float3 start, end;
};

class Material;

struct RayHit
{
	float t;
	float3 p;
	float3 normal;
	Material* mat_ptr;
};

class Hitable {
public:
	virtual bool Hit(const Ray& r, float t_min, float t_max, RayHit& rec) const = 0;
};



__device__ float3 MultiplyByScalar(const float3& multiplier, const float& t) {
	float3 output = multiplier;
	output.x *= t;
	output.y *= t;
	output.z *= t;
	return output;
}

__device__ float3 AddByScalar(const float3& multiplier, const float& t) {
	float3 output = multiplier;
	output.x += t;
	output.y += t;
	output.z += t;
	return output;
}


__device__ float3 AddVector(const float3& vector1, const float3& vector2) {
	float3 output = vector1;

	output.x += vector2.x;
	output.y += vector2.y;
	output.z += vector2.z;

	return output;
}

__device__ float3 MultiplyVector(const float3& vector1, const float3& vector2) {
	float3 output = vector1;

	output.x *= vector2.x;
	output.y *= vector2.y;
	output.z *= vector2.z;

	return output;
}

__device__ inline float3 UnitVector(const float3& vector) {
	float3 output;
	float length = vector.x + vector.y + vector.z;

	output.x = vector.x / length;
	output.y = vector.y / length;
	output.z = vector.z / length;

	return output;
}

__device__ inline float3 Color(const Ray& r, Hitable** world) {
	float3 output;

	float t = 0.5f * (UnitVector(r.Direction()).y + 1.0f);
	float3 intermediate1 = MultiplyByScalar(float3(1.0f, 1.0f, 1.0f), 1.0f - t);
	float3 intermediate2 = MultiplyByScalar(float3(0.5f, 0.7f, 1.0f), t);

	return AddVector(intermediate1, intermediate2);
}