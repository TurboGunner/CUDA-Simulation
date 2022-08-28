#pragma once

#include <vulkan/vulkan.h> //WIP!

#include <deque>
#include <functional>
#include <string>

using std::deque;
using std::function;
using std::string;

struct VulkanParameters {
	VkDevice device_;
};

struct AllocationParams {
	AllocationParams() = default;

	AllocationParams(function<void()> func_in, string name_in = "") {
		name_ = name_in;
		delete_func_ = func_in;
	}

	function<void()> delete_func_;
	string name_;
};

struct VulkanMemoryManager {

	void PushFunction(function<void()>& function, string name = "") {
		allocations_.push_back(AllocationParams(function, name));
	}

	void Flush() {
		for (auto it = allocations_.rbegin(); it != allocations_.rend(); it++) {
			(*it).delete_func_();
		}

		allocations_.clear();
	}
	deque<AllocationParams> allocations_;
};