use std::{ffi::CStr, sync::Arc};

use ash::vk;
use gfx_hal::{
    device::Device, error::GfxHalError, queue::Queue, surface::Surface, swapchain::Swapchain,
    swapchain::SwapchainConfig, sync::Fence, sync::Semaphore,
};
use gpu_allocator::{vulkan::Allocator, MemoryLocation};
use parking_lot::Mutex;
use resource_manager::{ImageHandle, ResourceManager, ResourceManagerError};
use thiserror::Error;
use tracing::{debug, error, info, warn};
 // Assuming winit is used by the app

// Re-export ash for convenience if needed elsewhere
pub use ash;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Debug, Error)]
pub enum RendererError {
    #[error("Graphics HAL Error: {0}")]
    GfxHal(#[from] GfxHalError),
    #[error("Resource Manager Error: {0}")]
    ResourceManager(#[from] ResourceManagerError),
    #[error("Egui Ash Renderer Error: {0}")]
    EguiRenderer(#[from] egui_ash_renderer::RendererError),
    #[error("Vulkan Error: {0}")]
    Vulkan(#[from] vk::Result),
    #[error("Failed to create shader module: {0}")]
    ShaderCreation(vk::Result),
    #[error("Failed to create pipeline layout: {0}")]
    PipelineLayoutCreation(vk::Result),
    #[error("Failed to create graphics pipeline: {0}")]
    PipelineCreation(vk::Result),
    #[error("Failed to create command pool: {0}")]
    CommandPoolCreation(vk::Result),
    #[error("Failed to allocate command buffers: {0}")]
    CommandBufferAllocation(vk::Result),
    #[error("Failed to begin command buffer: {0}")]
    CommandBufferBegin(vk::Result),
    #[error("Failed to end command buffer: {0}")]
    CommandBufferEnd(vk::Result),
    #[error("Swapchain acquisition failed")]
    SwapchainAcquisitionFailed,
    #[error("Swapchain is suboptimal")]
    SwapchainSuboptimal,
    #[error("Window reference is missing")] // If using raw-window-handle directly
    MissingWindow,
    #[error("Failed to get image info from resource manager")]
    ImageInfoUnavailable,
    #[error("Failed to get allocator from resource manager")]
    AllocatorUnavailable, // Added based on egui requirement
}

struct FrameData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: Semaphore,
    render_finished_semaphore: Semaphore,
    in_flight_fence: Fence,
}

struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

pub struct Renderer {
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    resource_manager: Arc<ResourceManager>,
    allocator: Arc<Mutex<Allocator>>, // Need direct access for egui

    surface: Arc<Surface>,        // Keep surface for recreation
    swapchain: Option<Swapchain>, // Option<> because it's recreated
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::SurfaceFormatKHR,
    swapchain_extent: vk::Extent2D,

    depth_image_handle: ImageHandle,
    depth_image_view: vk::ImageView, // Store the view directly
    depth_format: vk::Format,

    triangle_pipeline_layout: vk::PipelineLayout,
    triangle_pipeline: vk::Pipeline,

    frames_data: Vec<FrameData>,
    current_frame: usize,

    // Window state tracking (needed for recreation)
    window_resized: bool,
    current_width: u32,
    current_height: u32,
}

impl Renderer {
    pub fn new(
        instance: Arc<gfx_hal::instance::Instance>, // Needed for allocator
        device: Arc<Device>,
        graphics_queue: Arc<Queue>,
        surface: Arc<Surface>,
        resource_manager: Arc<ResourceManager>,
        initial_width: u32,
        initial_height: u32,
    ) -> Result<Self, RendererError> {
        info!("Initializing Renderer...");

        let allocator = resource_manager.allocator();

        let (swapchain, format, extent, image_views) = Self::create_swapchain_and_views(
            &device,
            &surface,
            initial_width,
            initial_height,
            None, // No old swapchain initially
        )?;

        let depth_format = Self::find_depth_format(&instance, &device)?;
        let (depth_image_handle, depth_image_view) =
            Self::create_depth_resources(&device, &resource_manager, extent, depth_format)?;

        let (triangle_pipeline_layout, triangle_pipeline) =
            Self::create_triangle_pipeline(&device, format.format, depth_format)?;

        let frames_data = Self::create_frame_data(&device)?;

        info!("Renderer initialized successfully.");

        Ok(Self {
            device,
            graphics_queue,
            resource_manager,
            allocator, // Store the allocator Arc
            surface,
            swapchain: Some(swapchain),
            swapchain_image_views: image_views,
            swapchain_format: format,
            swapchain_extent: extent,
            depth_image_handle,
            depth_image_view,
            depth_format,
            triangle_pipeline_layout,
            triangle_pipeline,
            frames_data,
            current_frame: 0,
            window_resized: false,
            current_width: initial_width,
            current_height: initial_height,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.window_resized = true;
            self.current_width = width;
            self.current_height = height;
            debug!("Window resize requested to {}x{}", width, height);
        } else {
            debug!("Ignoring resize to 0 dimensions");
        }
    }

    pub fn render_frame(&mut self) -> Result<(), RendererError> {
        // --- Handle Resize ---
        if self.window_resized {
            self.window_resized = false;
            debug!("Executing resize...");
            self.recreate_swapchain()?;
            // Skip rendering this frame as swapchain is new
            return Ok(());
        }

        // --- Wait for Previous Frame ---
        let frame_index = self.current_frame;
        let frame_data = &self.frames_data[frame_index];

        frame_data.in_flight_fence.wait(None)?; // Wait indefinitely

        // --- Acquire Swapchain Image ---
        let (image_index, suboptimal) = unsafe {
            // Need unsafe block for acquire_next_image
            self.swapchain
                .as_ref()
                .ok_or(RendererError::SwapchainAcquisitionFailed)? // Should exist
                .acquire_next_image(
                    u64::MAX, // Timeout
                    Some(&frame_data.image_available_semaphore),
                    None, // Don't need a fence here
                )?
        };

        if suboptimal {
            warn!("Swapchain is suboptimal, scheduling recreation.");
            self.window_resized = true; // Trigger recreation next frame
                                        // Reset fence *before* returning, otherwise we deadlock next frame
            frame_data.in_flight_fence.reset()?;
            return Ok(()); // Skip rendering
        }

        // --- Reset Fence (only after successful acquisition) ---
        frame_data.in_flight_fence.reset()?;

        // --- Record Command Buffer ---
        unsafe {
            // Need unsafe for Vulkan commands
            self.device
                .raw()
                .reset_command_pool(frame_data.command_pool, vk::CommandPoolResetFlags::empty())?;
        }

        let command_buffer = frame_data.command_buffer;
        let cmd_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            // Need unsafe for Vulkan commands
            self.device
                .raw()
                .begin_command_buffer(command_buffer, &cmd_begin_info)?;
        }

        // --- Dynamic Rendering Setup ---
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain_image_views[image_index as usize])
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.1, 1.0],
                },
            });

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_image_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE) // Or STORE if needed
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment))
            .depth_attachment(&depth_attachment);

        // --- Begin Dynamic Rendering ---
        unsafe {
            // Need unsafe for Vulkan commands
            self.device
                .raw()
                .cmd_begin_rendering(command_buffer, &rendering_info);
        }

        // --- Set Viewport & Scissor ---
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swapchain_extent.width as f32,
            height: self.swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };
        unsafe {
            // Need unsafe for Vulkan commands
            self.device
                .raw()
                .cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.device
                .raw()
                .cmd_set_scissor(command_buffer, 0, &[scissor]);
        }

        // --- Draw Triangle ---
        unsafe {
            // Need unsafe for Vulkan commands
            self.device.raw().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.triangle_pipeline,
            );
            // Draw 3 vertices, 1 instance, 0 first vertex, 0 first instance
            self.device.raw().cmd_draw(command_buffer, 3, 1, 0, 0);
        }

        // --- End Dynamic Rendering ---
        unsafe {
            // Need unsafe for Vulkan commands
            self.device.raw().cmd_end_rendering(command_buffer);
        }

        // --- End Command Buffer ---
        unsafe {
            // Need unsafe for Vulkan commands
            self.device.raw().end_command_buffer(command_buffer)?;
        }

        // --- Submit Command Buffer ---
        let wait_semaphores = [frame_data.image_available_semaphore.handle()];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [frame_data.render_finished_semaphore.handle()];
        let command_buffers = [command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        // assert_eq!(
        //     self.graphics_queue.device().raw().handle(), // Device from Queue
        //     self.device.raw().handle(),                  // Device stored in Renderer
        //     "Device handle mismatch between Renderer and Graphics Queue!"
        // );

        unsafe {
            // Need unsafe for queue submit
            self.graphics_queue.submit(
                self.device.raw(),
                &[submit_info],
                Some(&frame_data.in_flight_fence),
            )?;
        }

        // --- Present ---
        let swapchains = [self.swapchain.as_ref().unwrap().handle()]; // Safe unwrap after acquire
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let suboptimal_present = unsafe {
            // Need unsafe for queue_present
            self.swapchain
                .as_ref()
                .unwrap() // Safe unwrap
                .loader()
                .queue_present(self.graphics_queue.handle(), &present_info)
                .map_err(|e| {
                    // Handle VK_ERROR_OUT_OF_DATE_KHR specifically
                    if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
                        RendererError::SwapchainSuboptimal
                    } else {
                        RendererError::Vulkan(e)
                    }
                })? // Returns true if suboptimal
        };

        if suboptimal_present {
            warn!("Swapchain is suboptimal after present, scheduling recreation.");
            self.window_resized = true; // Trigger recreation next frame
        }

        // --- Advance Frame Counter ---
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    // --- Helper: Swapchain Recreation ---
    fn recreate_swapchain(&mut self) -> Result<(), RendererError> {
        info!("Recreating swapchain...");
        self.device.wait_idle()?; // Wait until device is idle

        // 1. Cleanup old resources
        self.cleanup_swapchain_resources(); // Destroys views, depth, old swapchain

        // 2. Create new resources
        let (new_swapchain, new_format, new_extent, new_image_views) =
            Self::create_swapchain_and_views(
                &self.device,
                &self.surface,
                self.current_width,
                self.current_height,
                self.swapchain.as_ref().map(|s| s.handle()), // Pass old handle
            )?;

        let (new_depth_handle, new_depth_view) = Self::create_depth_resources(
            &self.device,
            &self.resource_manager,
            new_extent,
            self.depth_format, // Keep the same depth format
        )?;

        // 3. Update Renderer state
        self.swapchain = Some(new_swapchain);
        self.swapchain_format = new_format;
        self.swapchain_extent = new_extent;
        self.swapchain_image_views = new_image_views;
        self.depth_image_handle = new_depth_handle;
        self.depth_image_view = new_depth_view;

        // 4. Update Egui Renderer (if necessary, depends on its implementation)
        // It might need the new extent or recreate internal resources.
        // Assuming it handles extent changes via update_screen_descriptor called earlier.

        info!(
            "Swapchain recreated successfully ({}x{}).",
            new_extent.width, new_extent.height
        );
        Ok(())
    }

    // --- Helper: Cleanup Swapchain Dependent Resources ---
    fn cleanup_swapchain_resources(&mut self) {
        debug!("Cleaning up swapchain resources...");
        // Destroy depth buffer view
        unsafe {
            self.device
                .raw()
                .destroy_image_view(self.depth_image_view, None);
        }
        // Destroy depth buffer image via resource manager
        if let Err(e) = self.resource_manager.destroy_image(self.depth_image_handle) {
            error!("Failed to destroy depth image: {}", e);
            // Continue cleanup even if this fails
        }
        // Drop the old swapchain object (RAII in gfx_hal::Swapchain handles vkDestroySwapchainKHR)
        self.swapchain = None;
        debug!("Swapchain resources cleaned up.");
    }

    // --- Helper: Create Swapchain ---
    fn create_swapchain_and_views(
        device: &Arc<Device>,
        surface: &Arc<Surface>,
        width: u32,
        height: u32,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<
        (
            Swapchain,
            vk::SurfaceFormatKHR,
            vk::Extent2D,
            Vec<vk::ImageView>,
        ),
        RendererError,
    > {
        let details = Self::query_swapchain_support(device.physical_device_handle(), surface)?;

        let surface_format = Self::choose_swapchain_format(&details.formats);
        let present_mode = Self::choose_swapchain_present_mode(&details.present_modes);
        let extent = Self::choose_swapchain_extent(&details.capabilities, width, height);

        let mut image_count = details.capabilities.min_image_count + 1;
        if details.capabilities.max_image_count > 0
            && image_count > details.capabilities.max_image_count
        {
            image_count = details.capabilities.max_image_count;
        }

        let config = SwapchainConfig {
            desired_format: surface_format,
            desired_present_mode: present_mode,
            desired_image_count: image_count,
            extent,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            pre_transform: details.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        };

        let swapchain =
            unsafe { Swapchain::new(device.clone(), surface.clone(), config, old_swapchain)? };

        // Create Image Views
        let image_views = swapchain
            .image_views() // Assuming Swapchain::new creates and stores these
            .to_vec(); // Clone the slice into a Vec

        // If Swapchain::new doesn't create views, we need to do it here:
        /*
        let images = swapchain.images()?; // Assuming this method exists
        let mut image_views = Vec::with_capacity(images.len());
        for &image in images.iter() {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let view = unsafe { device.raw().create_image_view(&create_info, None)? };
            image_views.push(view);
        }
        */

        Ok((swapchain, surface_format, extent, image_views))
    }

    // --- Helper: Create Depth Resources ---
    fn create_depth_resources(
        device: &Arc<Device>,
        resource_manager: &Arc<ResourceManager>,
        extent: vk::Extent2D,
        depth_format: vk::Format,
    ) -> Result<(ImageHandle, vk::ImageView), RendererError> {
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(depth_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = resource_manager.create_image(&image_create_info, MemoryLocation::GpuOnly)?;

        // Get the vk::Image handle to create the view
        let image_info = resource_manager.get_image_info(handle)?;

        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image_info.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { device.raw().create_image_view(&view_create_info, None)? };

        Ok((handle, view))
    }

    // --- Helper: Create Triangle Pipeline ---
    fn create_triangle_pipeline(
        device: &Arc<Device>,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline), RendererError> {
        // --- Shaders (Hardcoded example) ---
        // Vertex Shader (GLSL) - outputs clip space position based on vertex index
        /*
        #version 450
        vec2 positions[3] = vec2[](
            vec2(0.0, -0.5),
            vec2(0.5, 0.5),
            vec2(-0.5, 0.5)
        );
        void main() {
            gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
        }
        */
        // Fragment Shader (GLSL) - outputs solid orange
        /*
        #version 450
        layout(location = 0) out vec4 outColor;
        void main() {
            outColor = vec4(1.0, 0.5, 0.0, 1.0); // Orange
        }
        */

        // Load compiled SPIR-V (replace with actual loading)
        let vert_shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vert.glsl.spv")); // Placeholder path
        let frag_shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/frag.glsl.spv")); // Placeholder path

        let vert_module = Self::create_shader_module(device, vert_shader_code)?;
        let frag_module = Self::create_shader_module(device, frag_shader_code)?;

        let main_function_name = CStr::from_bytes_with_nul(b"main\0").unwrap();

        let vert_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(main_function_name);

        let frag_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(main_function_name);

        let shader_stages = [vert_stage_info, frag_stage_info];

        // --- Fixed Function State ---
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default(); // No vertex buffers/attributes

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1) // Dynamic viewport
            .scissor_count(1); // Dynamic scissor

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE) // Draw front face
            .front_face(vk::FrontFace::CLOCKWISE) // Doesn't matter for hardcoded triangle
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false); // No blending for opaque triangle

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // --- Pipeline Layout ---
        let layout_info = vk::PipelineLayoutCreateInfo::default(); // No descriptors/push constants
        let pipeline_layout = unsafe {
            device
                .raw()
                .create_pipeline_layout(&layout_info, None)
                .map_err(RendererError::PipelineLayoutCreation)?
        };

        // --- Dynamic Rendering Info ---
        let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(std::slice::from_ref(&color_format))
            .depth_attachment_format(depth_format);

        // --- Graphics Pipeline ---
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            // No render pass needed with dynamic rendering!
            .push_next(&mut pipeline_rendering_info); // Chain dynamic rendering info

        let pipeline = unsafe {
            device
                .raw()
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| RendererError::PipelineCreation(e))?[0] // Get the first pipeline from the result Vec
        };

        // --- Cleanup Shader Modules ---
        unsafe {
            device.raw().destroy_shader_module(vert_module, None);
            device.raw().destroy_shader_module(frag_module, None);
        }

        Ok((pipeline_layout, pipeline))
    }

    fn create_shader_module(
        device: &Arc<Device>,
        code: &[u8],
    ) -> Result<vk::ShaderModule, RendererError> {
        // 1. Check if byte count is a multiple of 4 (Vulkan requirement)
        let byte_count = code.len();
        if byte_count == 0 {
            // Handle empty shader code case if necessary, maybe return error
            return Err(RendererError::ShaderCreation(
                vk::Result::ERROR_INITIALIZATION_FAILED,
            )); // Or a custom error
        }
        if byte_count % 4 != 0 {
            // This indicates an invalid SPIR-V file was loaded.
            // Panicking here is reasonable during development, or return a specific error.
            error!(
            "Shader code size ({}) is not a multiple of 4 bytes! Check the .spv file generation.",
            byte_count
        );
            // You could return an error instead of panicking:
            return Err(RendererError::ShaderCreation(
                vk::Result::ERROR_INITIALIZATION_FAILED,
            )); // Or a custom error like InvalidShaderData
                // panic!(
                //     "Shader code size ({}) is not a multiple of 4 bytes!",
                //     byte_count
                // );
        }

        // --- Alternative: Copying to Vec<u32> (Safest, but allocates/copies) ---
        let code_u32: Vec<u32> = code
            .chunks_exact(4)
            .map(|chunk| {
                u32::from_ne_bytes(chunk.try_into().expect("Chunk size is guaranteed to be 4"))
            }) // Use from_le_bytes if SPIR-V endianness matters (it's LE)
            .collect();
        let code_slice_ref = &code_u32; // Use this slice below
                                        // --------------------------------------------------------------------

        // 3. Create the shader module
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code_slice_ref); // Pass the &[u32] slice

        unsafe {
            device
                .raw()
                .create_shader_module(&create_info, None)
                .map_err(|e| {
                    error!("Failed to create shader module: {:?}", e); // Add logging
                    RendererError::ShaderCreation(e)
                })
        }
    }

    // --- Helper: Create Frame Sync Objects & Command Resources ---
    fn create_frame_data(device: &Arc<Device>) -> Result<Vec<FrameData>, RendererError> {
        let mut frames_data = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = Semaphore::new(device.clone())?;
            let render_finished_semaphore = Semaphore::new(device.clone())?;
            let in_flight_fence = Fence::new(device.clone(), true)?; // Create signaled

            // Create Command Pool
            let pool_info = vk::CommandPoolCreateInfo::default()
                .flags(
                    vk::CommandPoolCreateFlags::TRANSIENT
                        | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ) // Allow resetting individual buffers
                .queue_family_index(device.graphics_queue_family_index());
            let command_pool = unsafe {
                device
                    .raw()
                    .create_command_pool(&pool_info, None)
                    .map_err(RendererError::CommandPoolCreation)?
            };

            // Allocate Command Buffer
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = unsafe {
                device
                    .raw()
                    .allocate_command_buffers(&alloc_info)
                    .map_err(RendererError::CommandBufferAllocation)?[0]
            };

            frames_data.push(FrameData {
                command_pool,
                command_buffer, // Stays allocated, just reset/rerecorded
                image_available_semaphore,
                render_finished_semaphore,
                in_flight_fence,
            });
        }
        Ok(frames_data)
    }

    // --- Swapchain Support Helpers --- (Simplified versions)
    fn query_swapchain_support(
        physical_device: vk::PhysicalDevice,
        surface: &Arc<Surface>,
    ) -> Result<SwapchainSupportDetails, GfxHalError> {
        unsafe {
            let capabilities = surface.get_physical_device_surface_capabilities(physical_device)?;
            let formats = surface.get_physical_device_surface_formats(physical_device)?;
            let present_modes =
                surface.get_physical_device_surface_present_modes(physical_device)?;
            Ok(SwapchainSupportDetails {
                capabilities,
                formats,
                present_modes,
            })
        }
    }

    fn choose_swapchain_format(available_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        available_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB // Prefer SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0]) // Fallback to first available
            .clone()
    }

    fn choose_swapchain_present_mode(available_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        available_modes
            .iter()
            .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX) // Prefer Mailbox (low latency)
            .unwrap_or(&vk::PresentModeKHR::FIFO) // Guaranteed fallback
            .clone()
    }

    fn choose_swapchain_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window_width: u32,
        window_height: u32,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            // Window manager dictates extent
            capabilities.current_extent
        } else {
            // We can choose extent within bounds
            vk::Extent2D {
                width: window_width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: window_height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    // --- Helper: Find Depth Format ---
    fn find_depth_format(
        instance: &Arc<gfx_hal::instance::Instance>,
        device: &Arc<Device>,
    ) -> Result<vk::Format, RendererError> {
        let candidates = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        for &format in candidates.iter() {
            let props = unsafe {
                instance
                    .ash_instance()
                    .get_physical_device_format_properties(device.physical_device_handle(), format)
            };
            if props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            {
                return Ok(format);
            }
        }
        Err(RendererError::Vulkan(
            vk::Result::ERROR_FORMAT_NOT_SUPPORTED,
        )) // Or custom error
    }
}

// --- Drop Implementation ---
impl Drop for Renderer {
    fn drop(&mut self) {
        info!("Dropping Renderer...");
        // Ensure GPU is idle before destroying anything
        if let Err(e) = self.device.wait_idle() {
            error!("Error waiting for device idle during drop: {}", e);
            // Continue cleanup regardless
        }

        // Cleanup swapchain resources (views, depth buffer, swapchain object)
        self.cleanup_swapchain_resources();

        // Drop egui renderer explicitly before allocator/device go away
        // Assuming egui_renderer has a drop impl that cleans its Vulkan resources
        // std::mem::drop(self.egui_renderer); // Not needed if it implements Drop

        // Destroy pipelines
        unsafe {
            self.device
                .raw()
                .destroy_pipeline(self.triangle_pipeline, None);
            self.device
                .raw()
                .destroy_pipeline_layout(self.triangle_pipeline_layout, None);
        }

        // Destroy frame data (fences, semaphores, command pools)
        // Fences/Semaphores are handled by gfx_hal::Drop
        // Command buffers are freed with the pool
        for frame_data in self.frames_data.drain(..) {
            unsafe {
                self.device
                    .raw()
                    .destroy_command_pool(frame_data.command_pool, None);
            }
        }

        // Arcs (device, queue, resource_manager, surface, allocator) will drop automatically.
        // ResourceManager's Drop impl should handle allocator destruction if needed.
        info!("Renderer dropped.");
    }
}
