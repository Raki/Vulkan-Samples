/* Copyright (c) 2024, Huawei Technologies Co., Ltd.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "common/vk_common.h"
#include "common/vk_initializers.h"
#include "core/instance.h"
#include "platform/application.h"

namespace Utility
{

};

/**
 * @brief A self-contained (minimal use of framework) sample that illustrates
 * the rendering of a triangle
 */
class HelloTriangleV13 : public vkb::Application
{
	// Define the Vertex structure
	struct Vertex
	{
		glm::vec3 position;
		glm::vec3 normal;
	};

	struct InstanceData
	{
		glm::vec3 trans;
		glm::vec3 color;
		//float     rot;
	};

	struct UboScene
	{
		glm::mat4 proj;
	};

	// Define the vertex data
	std::vector<Vertex> vertices = {
	    {{50.0f, -50.f,0}, {0.0f, 0.0f, 1.0f}},        // Vertex 1: Red
	    {{50.f, 50.f,0}, {0.0f, 0.0f, 1.0f}},         // Vertex 2: Green
	    {{-50.f, 50.f,0}, {0.0f, 0.0f, 1.0f}},         // Vertex 3: Blue
	    {{-50.f, -50.f, 0}, {0.0f, 0.0f, 1.0f}}          // Vertex 3: Blue
	};

	const std::vector<uint32_t> indices = {0, 1, 2,0,2,3};

	std::vector<Vertex> cube_vertices = {
	    // Front face
	    {{-50.0f, -50.0f, 50.0f}, {0.0f, 0.0f, 1.0f}},        // Bottom left
	    {{50.0f, -50.0f, 50.0f}, {0.0f, 0.0f, 1.0f}},         // Bottom right
	    {{50.0f, 50.0f, 50.0f}, {0.0f, 0.0f, 1.0f}},          // Top right
	    {{-50.0f, 50.0f, 50.0f}, {0.0f, 0.0f, 1.0f}},         // Top left

	    // Back face
	    {{-50.0f, -50.0f, -50.0f}, {0.0f, 0.0f, -1.0f}},        // Bottom left
	    {{50.0f, -50.0f, -50.0f}, {0.0f, 0.0f, -1.0f}},         // Bottom right
	    {{50.0f, 50.0f, -50.0f}, {0.0f, 0.0f, -1.0f}},          // Top right
	    {{-50.0f, 50.0f, -50.0f}, {0.0f, 0.0f, -1.0f}},         // Top left

	    // Left face
	    {{-50.0f, -50.0f, -50.0f}, {-1.0f, 0.0f, 0.0f}},        // Bottom back
	    {{-50.0f, -50.0f, 50.0f}, {-1.0f, 0.0f, 0.0f}},         // Bottom front
	    {{-50.0f, 50.0f, 50.0f}, {-1.0f, 0.0f, 0.0f}},          // Top front
	    {{-50.0f, 50.0f, -50.0f}, {-1.0f, 0.0f, 0.0f}},         // Top back

	    // Right face
	    {{50.0f, -50.0f, -50.0f}, {1.0f, 0.0f, 0.0f}},        // Bottom back
	    {{50.0f, -50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}},         // Bottom front
	    {{50.0f, 50.0f, 50.0f}, {1.0f, 0.0f, 0.0f}},          // Top front
	    {{50.0f, 50.0f, -50.0f}, {1.0f, 0.0f, 0.0f}},         // Top back

	    // Top face
	    {{-50.0f, 50.0f, -50.0f}, {0.0f, 1.0f, 0.0f}},        // Back left
	    {{50.0f, 50.0f, -50.0f}, {0.0f, 1.0f, 0.0f}},         // Back right
	    {{50.0f, 50.0f, 50.0f}, {0.0f, 1.0f, 0.0f}},          // Front right
	    {{-50.0f, 50.0f, 50.0f}, {0.0f, 1.0f, 0.0f}},         // Front left

	    // Bottom face
	    {{-50.0f, -50.0f, -50.0f}, {0.0f, -1.0f, 0.0f}},        // Back left
	    {{50.0f, -50.0f, -50.0f}, {0.0f, -1.0f, 0.0f}},         // Back right
	    {{50.0f, -50.0f, 50.0f}, {0.0f, -1.0f, 0.0f}},          // Front right
	    {{-50.0f, -50.0f, 50.0f}, {0.0f, -1.0f, 0.0f}}          // Front left
	};

	std::vector<uint32_t> cube_indices = {
	    // Front face
	    0, 1, 2, 0, 2, 3,
	    // Back face
	    4, 5, 6, 4, 6, 7,
	    // Left face
	    8, 9, 10, 8, 10, 11,
	    // Right face
	    12, 13, 14, 12, 14, 15,
	    // Top face
	    16, 17, 18, 16, 18, 19,
	    // Bottom face
	    20, 21, 22, 20, 22, 23};

	struct Buffer
	{
		VkBuffer buffer = VK_NULL_HANDLE;
		VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
		void          *data = nullptr;
		size_t         count = 0;
	};

	struct Entity
	{

	};

	/**
	 * @brief Swapchain state
	 */
	struct SwapchainDimensions
	{
		/// Width of the swapchain.
		uint32_t width = 0;

		/// Height of the swapchain.
		uint32_t height = 0;

		/// Pixel format of the swapchain.
		VkFormat format = VK_FORMAT_UNDEFINED;
	};

	/**
	 * @brief Per-frame data
	 */
	struct PerFrame
	{
		VkFence         queue_submit_fence          = VK_NULL_HANDLE;
		VkCommandPool   primary_command_pool        = VK_NULL_HANDLE;
		VkCommandBuffer primary_command_buffer      = VK_NULL_HANDLE;
		VkSemaphore     swapchain_acquire_semaphore = VK_NULL_HANDLE;
		VkSemaphore     swapchain_release_semaphore = VK_NULL_HANDLE;
	};

	/**
	 * @brief Vulkan objects and global state
	 */
	struct Context
	{
		/// The Vulkan instance.
		VkInstance instance = VK_NULL_HANDLE;

		/// The Vulkan physical device.
		VkPhysicalDevice gpu = VK_NULL_HANDLE;

		/// The Vulkan device.
		VkDevice device = VK_NULL_HANDLE;

		/// The Vulkan device queue.
		VkQueue queue = VK_NULL_HANDLE;

		/// The swapchain.
		VkSwapchainKHR swapchain = VK_NULL_HANDLE;

		/// The swapchain dimensions.
		SwapchainDimensions swapchain_dimensions;

		/// The surface we will render to.
		VkSurfaceKHR surface = VK_NULL_HANDLE;

		/// The queue family index where graphics work will be submitted.
		int32_t graphics_queue_index = -1;

		/// The image view for each swapchain image.
		std::vector<VkImageView> swapchain_image_views;

		/// The handles to the images in the swapchain.
		std::vector<VkImage> swapchain_images;

		/// The graphics pipeline.
		VkPipeline pipeline = VK_NULL_HANDLE;

		/**
		 * The pipeline layout for resources.
		 * Not used in this sample, but we still need to provide a dummy one.
		 */
		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

		/// The debug utility messenger callback.
		VkDebugUtilsMessengerEXT debug_callback = VK_NULL_HANDLE;

		/// A set of semaphores that can be reused.
		std::vector<VkSemaphore> recycled_semaphores;

		/// A set of per-frame data.
		std::vector<PerFrame> per_frame;

		/// The Vulkan buffer object that holds the vertex data for the triangle.
		VkBuffer vertex_buffer = VK_NULL_HANDLE;

		/// The device memory allocated for the vertex buffer.
		VkDeviceMemory vertex_buffer_memory = VK_NULL_HANDLE;

		std::shared_ptr<Buffer> vBuff = nullptr, iBuff=nullptr, uBuff=nullptr,uBuff2=nullptr,
			instBuff=nullptr;

		VkDescriptorPool      descriptor_pool = VK_NULL_HANDLE;
		VkDescriptorSet       descriptor_set  = VK_NULL_HANDLE;
		VkDescriptorSet       descriptor_set_2 = VK_NULL_HANDLE;
		VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;

		glm::mat4 projection,tri1Mat,tri2Mat;
	};

  public:
	HelloTriangleV13() = default;

	virtual ~HelloTriangleV13();

	virtual bool prepare(const vkb::ApplicationOptions &options) override;

	virtual void update(float delta_time) override;

	virtual bool resize(const uint32_t width, const uint32_t height) override;

	bool validate_extensions(const std::vector<const char *>          &required,
	                         const std::vector<VkExtensionProperties> &available);

	bool validate_layers(const std::vector<const char *>      &required,
	                     const std::vector<VkLayerProperties> &available);

	void init_instance();

	void init_device();

	void init_vertex_buffer();

	void init_per_frame(PerFrame &per_frame);

	void teardown_per_frame(PerFrame &per_frame);

	void init_swapchain();

	VkShaderModule load_shader_module(const char *path, VkShaderStageFlagBits shader_stage);

	void init_pipeline();

	VkResult acquire_next_swapchain_image(uint32_t *image);

	void render_triangle_old(uint32_t swapchain_index);

	void render_triangle(uint32_t swapchain_index);

	VkResult present_image(uint32_t index);

	void transition_image_layout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask, VkPipelineStageFlags2 srcStage, VkPipelineStageFlags2 dstStage);

	uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties);

	template <typename T>
	std::shared_ptr<Buffer> create_buffer(const std::vector<T> &arr,const VkBufferUsageFlags& usage);

	void setup_descriptor_pool();
	void setup_descriptor_set_layout();
	void setup_descriptor_set();

	template <typename T>
	void update_descriptor_set(const std::shared_ptr<Buffer> buff,const std::vector<T> &arr);

  private:
	Context context;

	std::unique_ptr<vkb::Instance> vk_instance;
};

std::unique_ptr<vkb::Application> create_hello_triangle_1_3();

template <typename T>
inline std::shared_ptr<HelloTriangleV13::Buffer> HelloTriangleV13::create_buffer(const std::vector<T> &arr, const VkBufferUsageFlags &usage)
{
	VkBuffer buffer;
	VkDeviceMemory buffer_memory;

	VkDeviceSize buffer_size = sizeof(arr[0]) * arr.size();

	// Create the vertex buffer
	VkBufferCreateInfo vertext_buffer_info{
	    .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	    .flags       = 0,
	    .size        = buffer_size,
	    .usage       = usage,
	    .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

	VK_CHECK(vkCreateBuffer(context.device, &vertext_buffer_info, nullptr, &buffer));

	// Get memory requirements
	VkMemoryRequirements memory_requirements;
	vkGetBufferMemoryRequirements(context.device, buffer, &memory_requirements);

	// Allocate memory for the buffer
	VkMemoryAllocateInfo alloc_info{
	    .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .allocationSize  = memory_requirements.size,
	    .memoryTypeIndex = find_memory_type(context.gpu, memory_requirements.memoryTypeBits,
	                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};

	VK_CHECK(vkAllocateMemory(context.device, &alloc_info, nullptr, &buffer_memory));

	// Bind the buffer with the allocated memory
	VK_CHECK(vkBindBufferMemory(context.device, buffer, buffer_memory, 0));

	// Map the memory and copy the vertex data
	void *data;
	VK_CHECK(vkMapMemory(context.device, buffer_memory, 0, buffer_size, 0, &data));
	memcpy(data, arr.data(), static_cast<size_t>(buffer_size));

	auto bufferObj    = std::make_shared<Buffer>();
	bufferObj->buffer = buffer;
	bufferObj->buffer_memory = buffer_memory;
	bufferObj->count         = arr.size();

	if (usage != VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
		vkUnmapMemory(context.device, buffer_memory);
	else
	{
		bufferObj->data = data;
	}

	return bufferObj;
}

template <typename T>
inline void HelloTriangleV13::update_descriptor_set(const std::shared_ptr<Buffer> buff, const std::vector<T> &arr)
{
	const VkDeviceSize buffer_size = sizeof(arr[0]) * arr.size();
	memcpy(buff->data, arr.data(), static_cast<size_t>(buffer_size));
}
