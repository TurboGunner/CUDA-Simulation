#include "axis_data_cuda.cuh"
#include <stdexcept>

AxisDataGPU::AxisDataGPU(Axis axis) {
	axis_ = axis;
	LoadDefaultDataSet();
}

AxisDataGPU::AxisDataGPU(unsigned int size, Axis axis) {
	if (size < 1) {
		throw std::invalid_argument("Error: Bounds must be at least greater than or equal to 1!");
	}
	size_ = size;
	axis_ = axis;
	LoadDefaultDataSet();
}

void AxisDataGPU::LoadDefaultDataSet() {
	unsigned int y_current = 0;
	for (y_current; y_current < size_; y_current++) {
		for (unsigned int i = 0; i < size_; i++) {
			map_.emplace(IndexPair(i, y_current), 0);
		}
	}
}

void AxisDataGPU::operator=(const AxisDataGPU& copy) {
	map_ = copy.map_;
	axis_ = copy.axis_;
	size_ = copy.size_;
}