#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fluid_sim_cuda.cuh"

#include "handler_methods.hpp"
#include "vector_field.hpp"
#include "fluid_sim.hpp"

#include <iostream>
#include <functional>

#include <stdio.h>

#include <openvdb/openvdb.h>

using std::vector;
using std::reference_wrapper;
using std::function;


void OpenVDBTest(FluidSim& sim) {
    VectorField velocity = sim.velocity_;

    openvdb::initialize();
    openvdb::FloatGrid::Ptr gridX = openvdb::FloatGrid::create(),
        gridY = openvdb::FloatGrid::create();

    openvdb::FloatGrid::Accessor accessorX = gridX->getAccessor(),
        accessorY = gridY->getAccessor();

    openvdb::Coord xyz(0, 0, 0);

    unsigned int size = (unsigned int)sqrt(velocity.GetVectorMap().size());
    unsigned int y_current = 0;

    for (y_current; y_current < size; y_current++) {
        for (unsigned int i = 0; i < size; i++) {
            IndexPair current(i, y_current);
            xyz.reset(i, y_current, 0);
            accessorX.setValue(xyz, velocity.GetVectorMap()[current].vx);
            accessorY.setValue(xyz, velocity.GetVectorMap()[current].vy);
            std::cout << velocity.GetVectorMap()[current].vy << std::endl;
        }
    }

    gridX->setName("VelocityVectorX");
    gridY->setName("VelocityVectorY");
    openvdb::io::File file("VelocityVector.vdb");
    openvdb::GridPtrVec grids;
    grids.push_back(gridX);
    grids.push_back(gridY);
    file.write(grids);
    file.close();
}

int main()
{
    unsigned int iter = 32, side_bound = 128;
    FluidSim simulation(.1f, 1.0f, 1, side_bound, side_bound, iter);

    cudaError_t cuda_status = cudaSuccess;

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "cudaSetDevice failed!", "main",
        cuda_status, "Do you have a CUDA-capable GPU installed?");

    float a_fac = simulation.dt_ * simulation.diffusion_ * (simulation.size_x_ - 2) * (simulation.size_x_ - 2);
    float c_fac = 1.0f + (4.0f * a_fac);

    simulation.Simulate();

    //CudaExceptionHandler(cuda_status, "LinearSolverCuda failed!");

    cuda_status = cudaDeviceReset();
    CudaExceptionHandler(cuda_status, "cudaDeviceReset failed!");

    OpenVDBTest(simulation);

    return 0;
}