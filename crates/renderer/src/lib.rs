use std::{
    collections::HashMap,
    ffi::c_void,
    mem,
    sync::{Arc, Mutex},
    time::Instant,
};

use ash::vk;
use egui::{ClippedPrimitive, TextureId, TexturesDelta};
use egui_ash_renderer::{DynamicRendering, Options, Renderer as EguiRenderer};
use gfx_hal::{
    device::Device, error::GfxHalError, queue::Queue, surface::Surface, swapchain::Swapchain,
    swapchain::SwapchainConfig, sync::Fence, sync::Semaphore,
};
use glam::{Mat4, Vec3};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation,
};
use resource_manager::{
    ImageHandle, Material, ResourceManager, ResourceManagerError, SamplerHandle, Texture,
};
use shared::{CameraInfo, UniformBufferObject};
use thiserror::Error;
use tracing::{debug, error, info, warn};

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const MAX_MATERIALS: usize = 150;

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
    #[error("Allocator Error: {0}")]
    AllocatorError(#[from] gpu_allocator::AllocationError),

    #[error("Other Error: {0}")]
    Other(String),
}

impl<T> From<std::sync::PoisonError<T>> for RendererError {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Self::AllocatorUnavailable
    }
}

struct FrameData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: Semaphore,
    render_finished_semaphore: Semaphore,
    textures_to_free: Option<Vec<TextureId>>,
    in_flight_fence: Fence,

    descriptor_set: vk::DescriptorSet,
    uniform_buffer_object: UniformBufferObject,
    uniform_buffer: vk::Buffer,
    uniform_buffer_allocation: Allocation,
    uniform_buffer_mapped_ptr: *mut c_void,
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

    scene: scene::Scene,

    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,

    material_descriptor_set_layout: vk::DescriptorSetLayout,

    egui_renderer: EguiRenderer,

    depth_image_handle: ImageHandle,
    depth_image_view: vk::ImageView, // Store the view directly
    depth_format: vk::Format,

    model_pipeline_layout: vk::PipelineLayout,
    model_pipeline: vk::Pipeline,

    material_descriptor_sets: HashMap<usize, vk::DescriptorSet>,

    default_white_texture: Option<Arc<Texture>>,
    default_sampler: SamplerHandle,

    frames_data: Vec<FrameData>,
    current_frame: usize,

    // Window state tracking (needed for recreation)
    window_resized: bool,
    current_width: u32,
    current_height: u32,

    start_time: Instant,
}

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        instance: Arc<gfx_hal::instance::Instance>, // Needed for allocator
        device: Arc<Device>,
        graphics_queue: Arc<Queue>,
        surface: Arc<Surface>,
        resource_manager: Arc<ResourceManager>,
        scene: scene::Scene,
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

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device)?;
        let material_descriptor_set_layout = Self::create_material_descriptor_set_layout(&device)?;

        let descriptor_set_layouts = [descriptor_set_layout, material_descriptor_set_layout];

        let descriptor_pool = Self::create_descriptor_pool(&device)?;

        let (model_pipeline_layout, model_pipeline) = Self::create_model_pipeline(
            &device,
            format.format,
            depth_format,
            &descriptor_set_layouts,
        )?;

        let start_time = Instant::now();

        let frames_data = Self::create_frame_data(
            &device,
            &resource_manager,
            descriptor_pool,
            &descriptor_set_layouts,
            swapchain.extent(),
        )?;

        info!("Renderer initialized successfully.");

        let egui_renderer = EguiRenderer::with_gpu_allocator(
            resource_manager.allocator(),
            device.raw().clone(),
            DynamicRendering {
                color_attachment_format: swapchain.format().format,
                depth_attachment_format: Some(depth_format),
            },
            Options {
                srgb_framebuffer: true,
                in_flight_frames: MAX_FRAMES_IN_FLIGHT,
                ..Default::default()
            },
        )?;

        let default_sampler = resource_manager.get_or_create_sampler(&Default::default())?;

        let default_white_texture = Some(Self::create_default_texture(
            device.clone(),
            resource_manager.clone(),
        ));

        Ok(Self {
            device,
            graphics_queue,
            resource_manager,
            egui_renderer,
            allocator, // Store the allocator Arc
            surface,
            swapchain: Some(swapchain),
            swapchain_image_views: image_views,
            swapchain_format: format,
            swapchain_extent: extent,
            descriptor_set_layout,
            descriptor_pool,

            material_descriptor_set_layout,
            depth_image_handle,
            depth_image_view,
            depth_format,
            model_pipeline_layout,
            model_pipeline,

            material_descriptor_sets: HashMap::new(),

            default_white_texture,
            default_sampler,

            frames_data,
            scene,
            current_frame: 0,
            window_resized: false,
            current_width: initial_width,
            current_height: initial_height,

            start_time,
        })
    }

    /// Gets or creates/updates a descriptor set for a given material.
    fn get_or_create_material_set(
        &mut self,
        material: &Arc<Material>, // Use Arc<Material> directly if hashable, or use a unique ID
    ) -> Result<vk::DescriptorSet, RendererError> {
        // Return generic error

        // Use a unique identifier for the material instance if Arc<Material> isn't directly hashable
        // or if pointer comparison isn't reliable across runs/reloads.
        // For simplicity here, we use the Arc's pointer address as a key.
        // WARNING: This is only safe if the Arc<Material> instances are stable!
        // A better key might be derived from material.name or a generated ID.
        let material_key = Arc::as_ptr(material) as usize;

        if let Some(set) = self.material_descriptor_sets.get(&material_key) {
            return Ok(*set);
        }

        // --- Allocate Descriptor Set ---
        let layouts = [self.material_descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_set = unsafe { self.device.raw().allocate_descriptor_sets(&alloc_info)? }[0];

        // --- Update Descriptor Set ---
        let (image_handle, view_handle, sampler_handle) = match &material.base_color_texture {
            Some(texture) => {
                // Get the default view handle associated with the image
                let img_info = self.resource_manager.get_image_info(texture.handle)?;
                let view_h = img_info.default_view_handle.ok_or(RendererError::Other(
                    "Image missing default view handle".to_string(),
                ))?;
                // Use the sampler specified by the material, or the default
                let sampler_h = material.base_color_sampler.unwrap_or(self.default_sampler);
                (texture.handle, view_h, sampler_h)
            }
            None => {
                // Use default white texture
                let default_tex =
                    self.default_white_texture
                        .as_ref()
                        .ok_or(RendererError::Other(
                            "Default texture not created".to_string(),
                        ))?;
                let img_info = self.resource_manager.get_image_info(default_tex.handle)?;
                let view_h = img_info.default_view_handle.ok_or(RendererError::Other(
                    "Default image missing default view handle".to_string(),
                ))?;
                (default_tex.handle, view_h, self.default_sampler)
            }
        };

        // Get the actual Vulkan handles
        let image_view_info = self.resource_manager.get_image_view_info(view_handle)?;
        let sampler_info = self.resource_manager.get_sampler_info(sampler_handle)?;

        let image_descriptor_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) // Expected layout for sampling
            .image_view(image_view_info.view) // The vk::ImageView
            .sampler(sampler_info.sampler); // The vk::Sampler

        let writes = [
            // Write for binding 0 (baseColorSampler)
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_descriptor_info)),
            // Add writes for other bindings (normal map, etc.) here
        ];

        unsafe {
            self.device.raw().update_descriptor_sets(&writes, &[]); // Update the set
        }

        // Store in cache
        self.material_descriptor_sets
            .insert(material_key, descriptor_set);
        Ok(descriptor_set)
    }

    fn create_default_texture(
        device: Arc<Device>, // Need device Arc for RM
        resource_manager: Arc<ResourceManager>,
    ) -> Arc<Texture> {
        let width = 1;
        let height = 1;
        let data = [255u8, 255, 255, 255]; // White RGBA
        let format = vk::Format::R8G8B8A8_UNORM; // Or SRGB if preferred

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let handle = resource_manager
            .create_image_init(
                &create_info,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageAspectFlags::COLOR,
                &data,
            )
            .expect("Failed to create default white texture");

        Arc::new(Texture {
            handle,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
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

    pub fn update_textures(&mut self, textures_delta: TexturesDelta) -> Result<(), RendererError> {
        tracing::trace!("Updating EGUI textures!");

        if !textures_delta.free.is_empty() {
            self.frames_data[self.current_frame].textures_to_free =
                Some(textures_delta.free.clone());
        }

        if !textures_delta.set.is_empty() {
            self.egui_renderer
                .set_textures(
                    self.device.get_graphics_queue().handle(),
                    self.frames_data[self.current_frame].command_pool,
                    textures_delta.set.as_slice(),
                )
                .expect("Failed to update texture");
        }

        Ok(())
    }

    pub fn render_frame(
        &mut self,
        pixels_per_point: f32,
        clipped_primitives: &[ClippedPrimitive],
        camera_info: CameraInfo,
    ) -> Result<(), RendererError> {
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
        let frame_data = &mut self.frames_data[frame_index];

        frame_data.in_flight_fence.wait(None)?; // Wait indefinitely

        // --- Acquire Swapchain Image ---
        let swapchain_ref = self
            .swapchain
            .as_ref()
            .ok_or(RendererError::SwapchainAcquisitionFailed)?;

        let (image_index, suboptimal) = unsafe {
            // Need unsafe block for acquire_next_image
            swapchain_ref.acquire_next_image(
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

        if let Some(textures) = frame_data.textures_to_free.take() {
            self.egui_renderer.free_textures(&textures)?;
        }

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

        // -- Update uniform buffer --
        self.update_uniform_buffer(camera_info)?;

        let frame_data = &mut self.frames_data[self.current_frame];
        let swapchain_ref = self
            .swapchain
            .as_ref()
            .ok_or(RendererError::SwapchainAcquisitionFailed)?;

        unsafe {
            // Need unsafe for Vulkan commands
            self.device
                .raw()
                .begin_command_buffer(command_buffer, &cmd_begin_info)?;
        }

        let current_swapchain_image = swapchain_ref.images()[image_index as usize];

        let initial_layout_transition_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty()) // No need to wait for writes from previous frame/present
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE) // Will be written as attachment
            .old_layout(vk::ImageLayout::UNDEFINED) // Assume undefined or present_src; UNDEFINED is safer
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) // Layout needed for rendering
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(current_swapchain_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            self.device.raw().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE, // Source stage (nothing before this)
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, // Destination stage (where write happens)
                vk::DependencyFlags::empty(),
                &[],                                  // No memory barriers
                &[],                                  // No buffer memory barriers
                &[initial_layout_transition_barrier], // The image barrier
            );
        }

        // --- Dynamic Rendering Setup ---
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain_image_views[image_index as usize])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
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

        unsafe {
            self.device.raw().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.model_pipeline,
            );

            self.device.raw().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.model_pipeline_layout,
                0,
                &[frame_data.descriptor_set],
                &[],
            );
        }

        let meshes = self.scene.meshes.clone();

        for mesh in meshes {
            let material_set = self.get_or_create_material_set(&mesh.material)?;

            unsafe {
                self.device.raw().cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.model_pipeline_layout,
                    1,
                    &[material_set],
                    &[],
                );
            }

            let model_matrix_bytes = unsafe {
                std::slice::from_raw_parts(
                    mesh.transform.as_ref().as_ptr() as *const u8,
                    std::mem::size_of::<Mat4>(),
                )
            };

            unsafe {
                self.device.raw().cmd_push_constants(
                    command_buffer,
                    self.model_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    model_matrix_bytes,
                );
            }

            mesh.geometry.draw(self.device.raw(), command_buffer)?;
        }

        let frame_data = &mut self.frames_data[self.current_frame];
        let swapchain_ref = self
            .swapchain
            .as_ref()
            .ok_or(RendererError::SwapchainAcquisitionFailed)?;

        tracing::trace!("Rendering EGUI");
        self.egui_renderer.cmd_draw(
            command_buffer,
            self.swapchain_extent,
            pixels_per_point,
            clipped_primitives,
        )?;
        tracing::trace!("Rendered EGUI");

        // --- End Dynamic Rendering ---
        unsafe {
            // Need unsafe for Vulkan commands
            self.device.raw().cmd_end_rendering(command_buffer);
        }

        let current_swapchain_image = swapchain_ref.images()[image_index as usize];

        let layout_transition_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(current_swapchain_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            self.device.raw().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[layout_transition_barrier],
            );
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
            swapchain_ref
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

        info!(
            "Swapchain recreated successfully ({}x{}).",
            new_extent.width, new_extent.height
        );
        Ok(())
    }

    // --- Helper: Cleanup Swapchain Dependent Resources ---
    fn cleanup_swapchain_resources(&mut self) {
        debug!("Cleaning up swapchain resources...");

        unsafe {
            self.device
                .raw()
                .destroy_image_view(self.depth_image_view, None);
        }
        // Destroy depth buffer image via resource manager
        if let Err(e) = self.resource_manager.destroy_image(self.depth_image_handle) {
            error!("Failed to destroy depth image: {}", e);
        }

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

        let image_views = swapchain
            .image_views() // Assuming Swapchain::new creates and stores these
            .to_vec(); // Clone the slice into a Vec

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

        let handle = resource_manager.create_image(
            &image_create_info,
            MemoryLocation::GpuOnly,
            vk::ImageAspectFlags::DEPTH,
        )?;

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
    fn create_model_pipeline(
        device: &Arc<Device>,
        color_format: vk::Format,
        depth_format: vk::Format,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<(vk::PipelineLayout, vk::Pipeline), RendererError> {
        // Load compiled SPIR-V (replace with actual loading)
        let vert_shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vert.glsl.spv")); // Placeholder path
        let frag_shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/frag.glsl.spv")); // Placeholder path

        let vert_module = Self::create_shader_module(device, vert_shader_code)?;
        let frag_module = Self::create_shader_module(device, frag_shader_code)?;

        let main_function_name = c"main";

        let vert_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(main_function_name);

        let frag_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(main_function_name);

        let shader_stages = [vert_stage_info, frag_stage_info];

        let binding_description = shared::Vertex::get_binding_decription();
        let attribute_descriptions = shared::Vertex::get_attribute_descriptions();

        // --- Fixed Function State ---
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(&attribute_descriptions);

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

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(mem::size_of::<Mat4>() as u32);

        // --- Pipeline Layout ---
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
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
        let create_info = vk::ShaderModuleCreateInfo::default().code(code_slice_ref); // Pass the &[u32] slice

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
    fn create_frame_data(
        device: &Arc<Device>,
        resource_manager: &Arc<ResourceManager>,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        swapchain_extent: vk::Extent2D,
    ) -> Result<Vec<FrameData>, RendererError> {
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

            tracing::info!("Allocated frame_data command_buffer: {:?}", command_buffer);

            let descriptor_set =
                Self::create_descriptor_set(device, descriptor_set_layouts, descriptor_pool)?;

            let (uniform_buffer, uniform_buffer_allocation, uniform_buffer_mapped_ptr) =
                Self::create_uniform_buffer(device, resource_manager)?;

            Self::update_descriptor_set(device.clone(), descriptor_set, uniform_buffer);

            let uniform_buffer_object = calculate_ubo(CameraInfo::default(), swapchain_extent);

            frames_data.push(FrameData {
                textures_to_free: None,
                command_pool,
                command_buffer, // Stays allocated, just reset/rerecorded
                image_available_semaphore,
                render_finished_semaphore,
                in_flight_fence,
                descriptor_set,
                uniform_buffer,
                uniform_buffer_allocation,
                uniform_buffer_mapped_ptr,
                uniform_buffer_object,
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
        *available_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB // Prefer SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
    }

    fn choose_swapchain_present_mode(available_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        *available_modes
            .iter()
            .find(|&&mode| mode == vk::PresentModeKHR::FIFO) // Prefer Mailbox (low latency)
            .unwrap_or(&vk::PresentModeKHR::FIFO)
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
        ))
    }

    fn create_material_descriptor_set_layout(
        device: &Arc<Device>,
    ) -> Result<vk::DescriptorSetLayout, RendererError> {
        let bindings = [
            // Binding 0: Combined Image Sampler (baseColorSampler)
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT), // Used in fragment shader
                                                              // Add more bindings here if needed (e.g., for normal map, metallic/roughness map)
                                                              // Binding 1: Uniform Buffer (Optional: for material factors)
                                                              // vk::DescriptorSetLayoutBinding::default()
                                                              //     .binding(1)
                                                              //     .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                                              //     .descriptor_count(1)
                                                              //     .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        Ok(unsafe {
            device
                .raw()
                .create_descriptor_set_layout(&layout_info, None)?
        })
    }

    fn create_descriptor_set_layout(
        device: &Arc<Device>,
    ) -> Result<vk::DescriptorSetLayout, RendererError> {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&ubo_layout_binding));

        let descriptor_set_layout = unsafe {
            device
                .raw()
                .create_descriptor_set_layout(&layout_info, None)?
        };

        Ok(descriptor_set_layout)
    }

    fn create_descriptor_pool(device: &Arc<Device>) -> Result<vk::DescriptorPool, RendererError> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: MAX_MATERIALS as u32,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32 + MAX_MATERIALS as u32);

        let descriptor_pool = unsafe { device.raw().create_descriptor_pool(&pool_info, None)? };

        Ok(descriptor_pool)
    }

    fn create_descriptor_set(
        device: &Arc<Device>,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        descriptor_pool: vk::DescriptorPool,
    ) -> Result<vk::DescriptorSet, RendererError> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(descriptor_set_layouts);

        let descriptor_set = unsafe { device.raw().allocate_descriptor_sets(&alloc_info)? }[0];

        Ok(descriptor_set)
    }

    fn create_uniform_buffer(
        device: &Arc<Device>,
        resource_manager: &Arc<ResourceManager>,
    ) -> Result<(vk::Buffer, Allocation, *mut std::ffi::c_void), RendererError> {
        let buffer_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation = resource_manager
            .allocator()
            .lock()?
            .allocate(&AllocationCreateDesc {
                name: "Uniform Buffer",
                requirements: unsafe {
                    {
                        let temp_buffer = device.raw().create_buffer(&buffer_info, None)?;
                        let req = device.raw().get_buffer_memory_requirements(temp_buffer);

                        device.raw().destroy_buffer(temp_buffer, None);
                        req
                    }
                },
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })?;

        let buffer = unsafe { device.raw().create_buffer(&buffer_info, None)? };
        tracing::info!("Created uniform buffer {:?}", buffer);

        unsafe {
            device
                .raw()
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        let mapped_ptr = allocation
            .mapped_ptr()
            .ok_or_else(|| {
                error!("Failed to get mapped pointer for CPU->GPU uniform buffer");
                ResourceManagerError::Other("Failed to map uniform buffer".to_string())
            })?
            .as_ptr();

        Ok((buffer, allocation, mapped_ptr))
    }

    fn update_descriptor_set(
        device: Arc<Device>,
        descriptor_set: vk::DescriptorSet,
        buffer: vk::Buffer,
    ) {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(mem::size_of::<UniformBufferObject>() as vk::DeviceSize);

        let descriptor_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe {
            device
                .raw()
                .update_descriptor_sets(std::slice::from_ref(&descriptor_write), &[]);
        }
    }

    fn update_uniform_buffer(&mut self, camera_info: CameraInfo) -> Result<(), RendererError> {
        let frame_data = &mut self.frames_data[self.current_frame];

        let ubo = calculate_ubo(camera_info, self.swapchain_extent);

        if frame_data.uniform_buffer_object != ubo {
            let ptr = frame_data.uniform_buffer_mapped_ptr;
            unsafe {
                let aligned_ptr = ptr as *mut UniformBufferObject;
                aligned_ptr.write(ubo);
            }
        }

        Ok(())
    }
}

fn calculate_ubo(camera_info: CameraInfo, swapchain_extent: vk::Extent2D) -> UniformBufferObject {
    let view = Mat4::look_at_rh(camera_info.camera_pos, camera_info.camera_target, Vec3::Y);

    let mut proj = Mat4::perspective_rh(
        camera_info.camera_fov.to_radians(),
        swapchain_extent.width as f32 / swapchain_extent.height as f32,
        0.1,
        1000.0,
    );

    proj.y_axis.y *= -1.0;

    UniformBufferObject { view, proj }
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
                .destroy_pipeline(self.model_pipeline, None);
            self.device
                .raw()
                .destroy_pipeline_layout(self.model_pipeline_layout, None);
        }

        unsafe {
            self.device
                .raw()
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .raw()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }

        // Destroy frame data (fences, semaphores, command pools)
        // Fences/Semaphores are handled by gfx_hal::Drop
        // Command buffers are freed with the pool
        for frame_data in self.frames_data.drain(..) {
            unsafe {
                self.device
                    .raw()
                    .destroy_buffer(frame_data.uniform_buffer, None);

                let mut allocator = self
                    .allocator
                    .lock()
                    .expect("Allocator Mutex to not be poisoned.");
                allocator
                    .free(frame_data.uniform_buffer_allocation)
                    .expect("Allocator to be able to free an allocation");
            }

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
