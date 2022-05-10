# Navier Stokes CUDA Simulator

## Introduction

A work in process program for the simulation of fluids via NVIDIA's CUDA.

Written in a mixture of CUDA C and standard C++ using the MSBuild compiler.

Uses a mixture of OOP, some functional patterns, and helper methods in order to reduce CUDA boilerplate with memory alloc/dealloc calls on GPU/system memory, as well as copying to and from GPU and system memory.

## CUDA Helper Structure

All relevant CUDA helper methods are contained within handler_methods.hpp/cpp.
> This is where the helper method that contains logic for wrapping common calls that each global kernel function indirectly or directly handles.

The helper class for managing allocation, deallocation, and copying of memory is located within handler_wrapper.hpp/cpp.
> This class makes many calls to handler_methods.cpp, and allows for more standard handling of the call as an object (i.e. with a proper destructor freeing both GPU and system memory at once).

Any relevant device functions (methods that are called from the GPU/from global kernel functions) are defined inline in cuda_sim_helpers.cuh.
> This includes calls within the global function for linear solving; as happens in projection.

## Data Structure

The primary data is stored within a Vectorfield; defined in vector_field.hpp/cpp.
> The data member that is contained is an unordered (hash) map, where the structure is:

```cplusplus
std::unordered_map<IndexPair, F_Vector, Hash> map_;
```

> Where the key is a struct called IndexPair, the value called F_Vector, and the passed in struct that contains the hash function logic is called Hash.

IndexPair is a struct that is defined in index_pair.hpp/cpp.
> It stores a coordinate system (x, y) stored as unsigned integers; and this allows for the ordered retrieval of F_Vector values while maintaining O(1) access times and lower cache utilization compared to an ordered map implementation.

F_Vector is a struct that is defined in f_vector.hpp/cpp.
> It stores the components (x, y) as floats, and this allows for the storing of multi-dimensional data as one struct. Also has methods for calculating magnitude with the provided data members.

## Fluid Simulation Implementation

A central struct called FluidSim; contained within fluid_sim.hpp/cpp holds the driving logic and handling of the fluid simulation.
>Velocity and the previous velocity are stored as VectorField structs.
>Density and the previous density are stored as AxisData structs.
>The constructor for FluidSim takes in these parameters:

```c++
FluidSim(float timestep, float diff, float visc, unsigned int size_x, unsigned int size_y, unsigned int iter);
```

As for the structure of the global kernel methods and accompanying helper methods, all of them are defined in one central CUDA header file called fluid_sim_cuda.cuh. This contains all relevant definitions for the CUDA calculations.

There are three main steps for fluid simulations that are defined for Navier-Stokes. They are:

### Diffusion
> Diffusion is not directly handled. Instead, it is defined internally with boundary conditions in the FluidSim struct, and then runs a linear solve based on the diffusion parameters. The linear solver definitions are defined in linear_solver.cu.

### Projection
> Projection is implemented in project.cu. It is implemented to be in accordance with the Helmholtz-Dodge Composition.

### Advection
> Advection is implemented in advect.cu. This governs the movement of the density and velocity throughout the field.


