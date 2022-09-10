# Renderer

The current renderer is implemented with Vulkan with LunarG validation layers. The input is handled by SDL, and the GUI is wrapped via IMGUI (experiment docking version).

## Organizational Architecture

The architecture is currently handled by one central header file (gui_driver.cuh). This essentially joins and binds all of the rendering logic together.

There are 6 files that inherent from this central header file:

- gui_display.cu
- imgui_initialization.cpp
- imgui_rendering.cpp
- vulkan_debugging.cpp
- vulkan_initialization.cpp
- vulkan_rendering.cpp

For organizational and discretization purposes, the most heavy initializations (i.e. pipelines, render passes) are helped via two types of classes. There are:

### Loader/Manager Classes

Loader/Manager classes are designated by the vast majority of logic being consolidated internally. The constructors often contain many parameters that are initialized through the primary file.

There are also many separated private methods that mainly serve as helper commands to initialize Vulkan parameter structs. Dependent on the purpose, they may also be base functions that initialize a single variable, and then are used in a root function that will initialize several of them.

They contain two methods in common as well. They are:

- Initialize()
    - > Is for the complete initialization of a given object/objects. Can contain input arguments for modes or other variables as well.
- Clean()
    - > Is for the complete destruction of a given object/objects. Typically will not require or accept any external arguments.

### Helper Classes

Helper classes are classes that mostly contain static methods, and either return or initialize references for the purposes of rendering. This was done due to many of the different classes sharing initialization logic (i.e. command buffer creation).

## Rendering Architecture

### Render Passes

Render passes are handled by rasterizer.cuh. This header file contains a class called RenderPassInitializer. Currently, it manages two separate attachments; one for the color, and one for the depth pass. There is currently one render pass that handles all rendering.

### Pipelines

Pipelines (both the pipeline and the layout) are handled by shader_loader.cuh. This header file contains a class called ShaderLoader. This contains one central pipeline with all the information required (vertex shaders, descriptor sets) for rendering. It also contains the color and depth attachments applicable to the pipeline.

### Swapchain

The swapchain is managed by swap_chain_manager.cuh. The file contains a class called SwapChainProperties. It handles swapchain images, swapchain image views, swapchain buffers, as well as the separate depth image and memory.

Swapchain reconstruction is handled by the method

```c++
void RecreateSwapChain(VkRenderPass& render_pass, VkCommandPool& command_pool);
```

Which takes in the render pass and the command pool as arguments to re-initialize the swapchain when it is deemed out of date.

### Synchronization

To increase performance and to decrease idling time, the method of employing multiple semaphores and fences is done.The file sync_structs.hpp is what contains the logic to initialize those objects.

The synchronization members are stored in vectors, as shown here as public data members:

```c++
vector<VkFence> fences_;
vector<VkSemaphore> render_semaphores_, present_semaphores_;
```

### Command Buffers

One time commands (i.e. IMGUI font loading) are handled by two helper methods defined in vulkan_helpers.hpp. 

BeginSingleTimeCommands is defined by:

```c++
static VkCommandBuffer BeginSingleTimeCommands(VkDevice& device, VkCommandPool command_pool, bool log = true);
```

EndSingleTimeCommands is defined by:

```c++
static VkResult EndSingleTimeCommands(VkCommandBuffer& command_buffer, VkDevice& device, VkCommandPool command_pool, VkQueue queue, const bool& log = true, const size_t& size = 1)
```

The command buffers that are used for rendering are also in vectors. The vector size for both synchronization and the command buffer are defined by the constant MAX_FRAMES_IN_FLIGHT_ defined by gui_driver.cuh.

The initialization method for the multiple command buffers is also contained in vulkan_helpers.hpp:

```c++
VkResult InitializeCommandBuffers(vector<VkCommandBuffer>& command_buffers, VkCommandPool& command_pool);
```

Where VkResult returns the success of the given command, the command_buffers vector is what is filled by InitializeCommandBuffers, and the command pool is used for the allocation of the command buffers.

## Mesh Architecture

### Vertex

The individual mesh data is handled via the struct Vertex, and Vertex is contained within vertex_data.hpp.

Vertex contains the following parameters.

```c++
glm::vec3 pos;
glm::vec3 normal;
glm::vec3 color;
```

Upon construction, it takes in the following arguments:

```c++
Vertex(float pos_x, float pos_y, float pos_z, float r, float g, float b, float n_x = 0.0f, float n_y = 0.0f, float n_z = 0.0f);
```

Where pos denotes position in 3D space, r, g, and b are used for intialization the color value, and the prefix n is referring to the normal of the given vertex in 3D space.

### Mesh Data Storage

Mesh data storage is handled by a class contained within vertex_data.hpp called MeshContainer. The initialization of it is handled by the constructor:

```c++
MeshContainer(const bool& collision_mode = false);
```

This contains a boolean called collision_mode, which is a default false value. Enabling this feature allows for the prevention of colliding vertices based on the position of a given vertex.

The private members that manage the storage and state of the vertices are:

```c++
vector<Vertex> vertices_;
unordered_map<glm::vec3, unsigned int, VectorHash> map_;
bool collision = false;
```

If collision is set to true via the constructor, it will exploit the hash collision properties of the unordered hash map to check if the size of the map has changed upon the entry of a new data point. This is done safely by try_emplace. If not, then it will warn via the logger of a collision.

The warning for intersecting vertices:

```c++
ProgramLog::OutputLine("Warning: Intersecting vertex!");
```

All of these members are private, and this in order to encapsulate the vector and map primarily. This allowed for the addition of safe methods, and custom logger warnings to be implemented upon an invalid input- whilst eliminating segfaulting or the requirement of program termination.

The warning for out of bounds:

```c++
ProgramLog::OutputLine("Warning: Out of bounds access on mesh container!");
```

### Mesh Data Management

Mesh management is handled by mesh_manager.hpp, and it contains the class VertexData. It handles the construction of initial vertices, helpers to bind to the render pipeline, and staging buffer management for the mesh. The scheme implemented for the purposes of performance to handle the mesh data were push constants; as opposed to heavily using uniform buffers.
