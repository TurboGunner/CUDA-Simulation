# Navier Stokes CUDA Simulator

## Introduction

A work in process program for the simulation of fluids via NVIDIA's CUDA.

Written in a mixture of CUDA C/C++ and standard C++ using the MSBuild compiler.

Data structures are handled via custom implementations, and file exports are handled via the OpenVDB library created by Dreamworks Animation.

Uses a mixture of OOP, some functional patterns, and helper methods in order to reduce CUDA boilerplate with memory alloc/dealloc calls on GPU/system memory, as well as copying to and from GPU and system memory.

## CUDA Helper Structure

All relevant CUDA helper methods are contained within handler_methods.hpp/cpp.
> This is where the helper method that contains logic for wrapping common calls that each global kernel function indirectly or directly handles.

## Data Structures

The VectorField data structure can be split into its constituent axes via the implementation of axis_data.hpp/cpp.
>It stores data in a hash map, where it stores a uni-dimensional float as the value. Also used in density; where the quantity is scalar.

```c++
HashMap<float>* map_;
```

> Where the key is the struct ndexPair, the value is a float primitive, and the passed in struct that contains the hash function logic is called HashDupe.

The multi-axis data (velocity) is stored within a VectorField; defined in vector_field.hpp/cpp.
> The data member that is contained is a pointer array of AxisData objects, where the structure is:

```c++
AxisData* map_;
```

IndexPair is a struct that is defined in index_pair.hpp/cpp.
> It stores a coordinate system (x, y) stored as unsigned integers; and this allows for the ordered retrieval of axis corresponding float values while maintaining O(1) access times and lower cache utilization compared to an ordered map implementation.

HashMap is a custom CUDA map implementation created in order to solve the issue of poor interopability with previous (un)abstracted code for the original data management implementation, as well as the amount of safeguards and additional code required to accomodate for the unpacking of VectorField and DataAxis to a float pointer, and the repacking back into the data structure.

It works in a standard host-device scheme- but differs as it is able to be accessed seamlessly by either device or host due to the setup of the data structure. This allows for memory to be accessed interchangably between the host and the device, and to eliminate the necessity for excessively bloated memory copy and allocation abstraction for the data structures; which significantly simplifies the process for which data management is done in both system and GPU contexts.

>Specific GitHub Page Here: https://github.com/TurboGunner/CUDAMap (this served as the basis for the original plan to use a hybridized scheme, but that was scrapped due to the inherent lack of support of asynchronous page migration required to make the unified memory viable performance wise except on Linux).

## Fluid Simulation Implementation

A central struct called FluidSim; contained within fluid_sim.hpp/cpp holds the driving logic and handling of the fluid simulation.
>Velocity and the previous velocity are stored as VectorField structs; which holds an AxisData struct for each axis.
>Density and the previous density are stored as AxisData structs.
>The constructor for FluidSim takes in these parameters:

```c++
FluidSim(float timestep, float diff, float visc, unsigned int size_x, unsigned int size_y, unsigned int iter, float time_max = 1);
```

As for the structure of the global kernel methods and accompanying helper methods, all of them are defined in one central CUDA header file called fluid_sim_cuda.cuh. This contains all relevant definitions for the CUDA calculations.

There are three main steps for fluid simulations that are defined for Navier-Stokes. They are:

### Diffusion
> Diffusion is not directly handled. Instead, it is defined internally with boundary conditions in the FluidSim struct, and then runs a linear solve based on the diffusion parameters. The linear solver definitions are defined in linear_solver.cu.

### Projection
> Projection is implemented in project.cu. It is implemented to be in accordance with the Helmholtz-Dodge Composition.

### Advection
> Advection is implemented in advect.cu. This governs the movement of the density and velocity throughout the field.


## To-do:

Move from Gauss-Siedel to a conjugate gradient solver for the linear systems solver to improve accuracy and quality of the projection and diffusion.

Move to 3D from 2D Navier-Stokes.

Further optimize memory allocation and usage via possible device pointers for the respective data structures for the fluid simulation.
