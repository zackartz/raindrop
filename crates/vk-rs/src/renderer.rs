use ash::{
    extensions::khr::{Surface, Swapchain},
    vk::{self, Buffer, DescriptorType},
    Device,
};
use egui::Vec2;
use egui_ash::EguiCommand;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use shaders_shared::{Material, PushConstants, UniformBufferObject};
use spirv_std::glam::{Mat4, Vec3, Vec4};
use std::{
    ffi::CString,
    mem::ManuallyDrop,
    panic,
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::texture_cache::TextureCache;

macro_rules! include_spirv {
    ($file:literal) => {{
        let bytes = include_bytes!($file);
        bytes
            .chunks_exact(4)
            .map(|x| x.try_into().unwrap())
            .map(match bytes[0] {
                0x03 => u32::from_le_bytes,
                0x07 => u32::from_be_bytes,
                _ => panic!("Unknown endianness"),
            })
            .collect::<Vec<u32>>()
    }};
}

pub struct DefaultTextures {
    white: Texture,              // Default albedo (white)
    metallic_roughness: Texture, // Default metallic-roughness (black metallic, 0.5 roughness)
    normal: Texture,             // Default normal map (flat normal)
    sampler: vk::Sampler,        // Common sampler for all textures
}

#[repr(C)]
#[derive(Debug, Clone)]
struct Vertex {
    position: Vec3,
    normal: Vec3,
    tex_coords: Vec2,
}
impl Vertex {
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(4 * 3)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(24)
                .build(),
        ]
    }
}

pub struct Texture {
    image: vk::Image,
    image_allocation: Option<Allocation>,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
}

impl Texture {
    fn new(
        device: &Device,
        allocator: Arc<Mutex<Allocator>>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Self {
        let mut allocator = allocator.lock().unwrap();

        let buffer_size = data.len() as u64;

        tracing::info!("DATA LEN: {}", buffer_size);
        let staging_buffer = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(buffer_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    None,
                )
                .expect("failed to create_buffer")
        };

        let staging_allocation = allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "staging_buffer",
                requirements: unsafe { device.get_buffer_memory_requirements(staging_buffer) },
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .expect("failed to allocate memory");

        unsafe {
            device
                .bind_buffer_memory(
                    staging_buffer,
                    staging_allocation.memory(),
                    staging_allocation.offset(),
                )
                .expect("failed to bind_buffer_memory");

            let ptr = staging_allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
            ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }

        let image = unsafe {
            device
                .create_image(
                    &vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_SRGB)
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .expect("failed to create image")
        };

        let image_allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "texture image",
                requirements: unsafe { device.get_image_memory_requirements(image) },
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .expect("failed to allocate memory");

        unsafe {
            device
                .bind_image_memory(image, image_allocation.memory(), image_allocation.offset())
                .expect("failed to bind image memory")
        };

        unsafe {
            let command_buffer = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("failed to allocate command_buffer")[0];

            device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("failed to begin_command_buffer");

            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier.build()],
            );

            let region = vk::BufferImageCopy::builder()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .buffer_offset(0)
                .buffer_row_length(width)
                .buffer_image_height(height);

            device.cmd_copy_buffer_to_image(
                command_buffer,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region.build()],
            );

            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier.build()],
            );

            device
                .end_command_buffer(command_buffer)
                .expect("failed to end the command buffer");

            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[command_buffer])
                        .build()],
                    vk::Fence::null(),
                )
                .expect("failed to submit queue");

            device.queue_wait_idle(queue).expect("failed to wait queue");

            device.free_command_buffers(command_pool, &[command_buffer]);

            device.destroy_buffer(staging_buffer, None);
            allocator
                .free(staging_allocation)
                .expect("failed to free memory");
        };

        let image_view = unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_SRGB)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
                .expect("failed to create image view")
        };

        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::builder()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT)
                        .anisotropy_enable(true)
                        .max_anisotropy(16.0)
                        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                        .unnormalized_coordinates(false)
                        .compare_enable(false)
                        .compare_op(vk::CompareOp::ALWAYS)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .mip_lod_bias(0.0)
                        .min_lod(0.0)
                        .max_lod(0.0),
                    None,
                )
                .expect("failed to create sampler")
        };

        Self {
            image,
            image_allocation: Some(image_allocation),
            image_view,
            sampler,
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.image_view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.image_allocation.take() {
            allocator.free(allocation).expect("failed to free memory");
        }
    }
}

pub struct Mesh {
    vertex_buffer: Buffer,
    vertex_buffer_allocation: Option<Allocation>,
    vertex_count: u32,
    transform: Mat4,
    albedo_texture: Option<Arc<Texture>>,
    metallic_roughness_texture: Option<Arc<Texture>>,
    normal_texture: Option<Arc<Texture>>,
    metallic_factor: f32,
    roughness_factor: f32,
    base_color: Vec4,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

pub struct Model {
    meshes: Vec<Mesh>,
    texture_cache: TextureCache,
}

impl Model {
    fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        for mesh in &mut self.meshes {
            unsafe {
                device.destroy_buffer(mesh.vertex_buffer, None);
                if let Some(vertex_buffer_allocation) = mesh.vertex_buffer_allocation.take() {
                    allocator
                        .free(vertex_buffer_allocation)
                        .expect("Failed to free memory");
                }
                self.texture_cache.cleanup(device, allocator);
            }
        }
    }
}

fn resize_texture(image: &image::DynamicImage, max_dimension: u32) -> image::DynamicImage {
    let width = image.width();
    let height = image.height();

    if width <= max_dimension && height <= max_dimension {
        return image.clone();
    }

    let aspect_ratio = width as f32 / height as f32;
    let (new_width, new_height) = if width > height {
        let w = max_dimension;
        let h = (w as f32 / aspect_ratio) as u32;
        (w, h)
    } else {
        let h = max_dimension;
        let w = (h as f32 * aspect_ratio) as u32;
        (w, h)
    };

    image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
}

fn load_texture_from_gltf(
    device: &Device,
    allocator: Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    texture: &gltf::Texture,
    buffers: &[gltf::buffer::Data],
    path: &Path,
) -> Option<Texture> {
    let img_data =
        gltf::image::Data::from_source(texture.source().source(), Some(path), buffers).ok()?;

    tracing::info!("WIDTH: {}", img_data.width);
    tracing::info!("HEIGHT: {}", img_data.height);

    // Resize texture if needed
    let image = image::DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(img_data.width, img_data.height, img_data.pixels).unwrap(),
    );

    const MAX_TEXTURE_DIMENSION: u32 = 2048; // Adjust this value based on your needs
    let resized = resize_texture(&image, MAX_TEXTURE_DIMENSION);

    let pixels_rgba = resized.to_rgba8().into_raw();

    Some(Texture::new(
        device,
        allocator,
        command_pool,
        queue,
        resized.width(),
        resized.height(),
        &pixels_rgba,
    ))
}

impl Model {
    fn load(
        device: &Device,
        allocator: Arc<Mutex<Allocator>>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        path: &str,
    ) -> Self {
        let path = Path::new(path);
        let base_path = path.parent();
        let (document, buffers, _) = gltf::import(path).expect("failed to load GLTF");
        let mut meshes = Vec::new();
        let mut texture_cache = TextureCache::new();

        for scene in document.scenes() {
            for node in scene.nodes() {
                meshes.extend(process_node(
                    node,
                    Mat4::IDENTITY,
                    &buffers,
                    device,
                    allocator.clone(),
                    command_pool,
                    queue,
                    base_path.unwrap(),
                    &mut texture_cache,
                ));
            }
        }

        Self {
            meshes,
            texture_cache,
        }
    }
}

fn process_node(
    node: gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    device: &Device,
    allocator: Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    path: &Path,
    texture_cache: &mut TextureCache,
) -> Vec<Mesh> {
    let mut meshes = Vec::new();

    let local_transform = {
        let (translation, rotation, scale) = node.transform().decomposed();
        Mat4::from_scale_rotation_translation(
            Vec3::from(scale),
            spirv_std::glam::Quat::from_array(rotation),
            Vec3::from(translation),
        )
    };
    let world_transform = parent_transform * local_transform;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let material = primitive.material();
            let pbr = material.pbr_metallic_roughness();

            let base_color = pbr.base_color_factor();
            let metallic_factor = pbr.metallic_factor();
            let roughness_factor = pbr.roughness_factor();

            let albedo_texture = pbr.base_color_texture().and_then(|tex| {
                texture_cache.get_or_load_texture(
                    format!("albedo_{:?}", tex.texture().source().source()),
                    || {
                        load_texture_from_gltf(
                            device,
                            allocator.clone(),
                            command_pool,
                            queue,
                            &tex.texture(),
                            buffers,
                            path,
                        )
                    },
                )
            });

            let metallic_roughness_texture = pbr.base_color_texture().and_then(|tex| {
                texture_cache.get_or_load_texture(
                    format!("mr_{:?}", tex.texture().source().source()),
                    || {
                        load_texture_from_gltf(
                            device,
                            allocator.clone(),
                            command_pool,
                            queue,
                            &tex.texture(),
                            buffers,
                            path,
                        )
                    },
                )
            });

            let normal_texture = pbr.base_color_texture().and_then(|tex| {
                texture_cache.get_or_load_texture(
                    format!("norm_{:?}", tex.texture().source().source()),
                    || {
                        load_texture_from_gltf(
                            device,
                            allocator.clone(),
                            command_pool,
                            queue,
                            &tex.texture(),
                            buffers,
                            path,
                        )
                    },
                )
            });

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            if let (Some(positions), Some(normals), Some(tex_coords)) = (
                reader.read_positions(),
                reader.read_normals(),
                reader.read_tex_coords(0),
            ) {
                let mut vertices = Vec::new();

                let indices = reader
                    .read_indices()
                    .map(|indices| indices.into_u32().collect::<Vec<_>>())
                    .unwrap_or_else(|| (0..positions.len() as u32).collect());

                let positions: Vec<_> = positions.collect();
                let normals: Vec<_> = normals.collect();
                let tex_coords: Vec<_> = tex_coords.into_f32().collect();

                for &index in indices.iter() {
                    let i = index as usize;
                    // Apply world transform to position
                    let position = world_transform.transform_point3(Vec3::new(
                        positions[i][0],
                        positions[i][1],
                        positions[i][2],
                    ));
                    let normal = world_transform.transform_vector3(Vec3::new(
                        normals[i][0],
                        normals[i][1],
                        normals[i][2],
                    ));

                    let vertex = Vertex {
                        position,
                        normal,
                        tex_coords: Vec2::new(tex_coords[i][0], tex_coords[i][1]),
                    };
                    vertices.push(vertex);
                }

                let (vertex_buffer, vertex_buffer_allocation) =
                    create_vertex_buffer(device, allocator.clone(), command_pool, queue, &vertices);

                meshes.push(Mesh {
                    vertex_buffer,
                    vertex_buffer_allocation: Some(vertex_buffer_allocation),
                    vertex_count: vertices.len() as u32,
                    albedo_texture,
                    metallic_roughness_texture,
                    normal_texture,
                    metallic_factor,
                    roughness_factor,
                    base_color: base_color.into(),
                    transform: Mat4::IDENTITY,
                    descriptor_sets: Vec::new(),
                });
            }
        }
    }

    for child in node.children() {
        meshes.extend(process_node(
            child,
            world_transform,
            buffers,
            device,
            allocator.clone(),
            command_pool,
            queue,
            path,
            texture_cache,
        ));
    }

    meshes
}

fn create_vertex_buffer(
    device: &Device,
    allocator: Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    vertices: &[Vertex],
) -> (Buffer, Allocation) {
    let mut allocator = allocator.lock().unwrap();

    let vertex_buffer_size = std::mem::size_of_val(vertices) as u64;

    let staging_buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(vertex_buffer_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
            .expect("failed to create buffer")
    };

    let staging_allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "Staging vertex buffer",
            requirements: unsafe { device.get_buffer_memory_requirements(staging_buffer) },
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })
        .expect("failed to allocate memory");

    unsafe {
        device
            .bind_buffer_memory(
                staging_buffer,
                staging_allocation.memory(),
                staging_allocation.offset(),
            )
            .expect("failed to bind buffer memory");

        let ptr = staging_allocation.mapped_ptr().unwrap().as_ptr() as *mut Vertex;
        ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
    }

    let vertex_buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(vertex_buffer_size)
                    .usage(
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                    ),
                None,
            )
            .expect("failed to create buffer")
    };

    let vertex_allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "Vertex Buffer",
            requirements: unsafe { device.get_buffer_memory_requirements(vertex_buffer) },
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })
        .expect("failed to allocate memory");

    unsafe {
        device
            .bind_buffer_memory(
                vertex_buffer,
                vertex_allocation.memory(),
                vertex_allocation.offset(),
            )
            .expect("failed to bind buffer memory");
    }

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .expect("failed to allocate command buffer")[0]
    };

    unsafe {
        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build(),
            )
            .expect("failed to begin command_buffer");

        device.cmd_copy_buffer(
            command_buffer,
            staging_buffer,
            vertex_buffer,
            &[vk::BufferCopy::builder().size(vertex_buffer_size).build()],
        );

        device
            .end_command_buffer(command_buffer)
            .expect("Failed to end command buffer");

        device
            .queue_submit(
                queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[command_buffer])
                    .build()],
                vk::Fence::null(),
            )
            .expect("failed to submit queue");

        device.queue_wait_idle(queue).expect("failed to wait queue");
        device.free_command_buffers(command_pool, &[command_buffer]);
    }

    unsafe {
        device.destroy_buffer(staging_buffer, None);
    }
    allocator
        .free(staging_allocation)
        .expect("failed to free memory");

    (vertex_buffer, vertex_allocation)
}

pub struct RendererInner {
    width: u32,
    height: u32,

    physical_device: vk::PhysicalDevice,
    device: Device,
    surface_loader: Surface,
    swapchain_loader: Swapchain,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    surface: vk::SurfaceKHR,
    queue: vk::Queue,

    swapchain: vk::SwapchainKHR,
    surface_format: vk::SurfaceFormatKHR,
    surface_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    uniform_buffers: Vec<Buffer>,
    uniform_buffer_allocations: Vec<Allocation>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    depth_images_and_allocations: Vec<(vk::Image, Allocation)>,
    color_image_views: Vec<vk::ImageView>,
    depth_image_views: Vec<vk::ImageView>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    model: Model,

    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_fences: Vec<vk::Fence>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    current_frame: usize,
    dirty_swapchain: bool,

    frame_counter: u32,
    camera_position: Vec3,
    camera_fov: f32,
    camera_yaw: f32,
    camera_pitch: f32,
    bg_color: Vec3,
    model_color: Vec3,
    accumulation_reset_needed: bool,
}

struct CreateFramebuffersResult(
    Vec<vk::Framebuffer>,
    Vec<(vk::Image, Allocation)>,
    Vec<vk::ImageView>,
    Vec<vk::ImageView>,
);

impl RendererInner {
    fn create_swapchain(
        physical_device: vk::PhysicalDevice,
        surface_loader: &ash::extensions::khr::Surface,
        swapchain_loader: &ash::extensions::khr::Swapchain,
        surface: vk::SurfaceKHR,
        queue_family_index: u32,
        width: u32,
        height: u32,
    ) -> (
        vk::SwapchainKHR,
        vk::SurfaceFormatKHR,
        vk::Extent2D,
        Vec<vk::Image>,
    ) {
        let surface_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .expect("Failed to get physical device surface formats")
        };
        let surface_format = *surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_UNORM
                    || format.format == vk::Format::R8G8B8A8_UNORM
            })
            .unwrap_or(&surface_formats[0]);
        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .expect("Failed to get physical device surface capabilities")
        };
        let surface_extent = if surface_capabilities.current_extent.width != u32::MAX {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width
                    .max(surface_capabilities.min_image_extent.width)
                    .min(surface_capabilities.max_image_extent.width),
                height: height
                    .max(surface_capabilities.min_image_extent.height)
                    .min(surface_capabilities.max_image_extent.height),
            }
        };

        let image_count = surface_capabilities.min_image_count + 1;
        let image_count = if surface_capabilities.max_image_count != 0 {
            image_count.min(surface_capabilities.max_image_count)
        } else {
            image_count
        };

        let image_sharing_mode = vk::SharingMode::EXCLUSIVE;
        let queue_family_indices = [queue_family_index];

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::MAILBOX)
            .clipped(true);

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create swapchain")
        };

        let swapchain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images")
        };

        (swapchain, surface_format, surface_extent, swapchain_images)
    }

    fn create_uniform_buffers<T: Sized>(
        device: &Device,
        allocator: Arc<Mutex<Allocator>>,
        swapchain_count: usize,
    ) -> (Vec<Buffer>, Vec<Allocation>) {
        let buffer_size = std::mem::size_of::<T>() as u64;

        let buffer_usage = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(buffer_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffers = (0..swapchain_count)
            .map(|_| unsafe {
                device
                    .create_buffer(&buffer_create_info, None)
                    .expect("Failed to create buffer")
            })
            .collect::<Vec<_>>();
        let buffer_memory_requirements =
            unsafe { device.get_buffer_memory_requirements(buffers[0]) };
        let buffer_alloc_info = gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Uniform Buffer",
            requirements: buffer_memory_requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        };
        let buffer_allocations = buffers
            .iter()
            .map(|_| {
                allocator
                    .lock()
                    .unwrap()
                    .allocate(&buffer_alloc_info)
                    .expect("Failed to allocate memory")
            })
            .collect::<Vec<_>>();
        for (&buffer, buffer_memory) in buffers.iter().zip(buffer_allocations.iter()) {
            unsafe {
                device
                    .bind_buffer_memory(buffer, buffer_memory.memory(), buffer_memory.offset())
                    .expect("Failed to bind buffer memory")
            }
        }

        (buffers, buffer_allocations)
    }

    fn create_descriptor_pool(device: &Device, total_sets_needed: usize) -> vk::DescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(total_sets_needed as u32)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(total_sets_needed as u32)
                .build(),
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(total_sets_needed as u32);

        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("failed to create descriptor pool")
        }
    }

    fn create_descriptor_set_layouts(
        device: &Device,
        swapchain_count: usize,
    ) -> Vec<vk::DescriptorSetLayout> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];

        let layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

        (0..swapchain_count)
            .map(|_| unsafe {
                device
                    .create_descriptor_set_layout(&layout_create_info, None)
                    .expect("Failed to create descriptor set layout")
            })
            .collect()
    }

    fn create_descriptor_sets(
        device: &Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        uniform_buffers: &[Buffer],
        allocator: Arc<Mutex<Allocator>>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        mesh: &Mesh,
    ) -> Vec<vk::DescriptorSet> {
        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(descriptor_set_layouts),
                )
                .expect("failed to allocate descriptor sets")
        };

        let default_texture =
            create_default_texture(device, allocator.clone(), command_pool, queue);
        let default_metallic_roughness = create_default_metallic_roughness_texture(
            device,
            allocator.clone(),
            command_pool,
            queue,
        );
        let default_normal =
            create_default_normal_texture(device, allocator.clone(), command_pool, queue);

        for (index, &descriptor_set) in descriptor_sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffers[index])
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build();

            let albedo_texture = mesh.albedo_texture.as_ref().unwrap_or(&default_texture);
            let albedo_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(albedo_texture.image_view)
                .sampler(albedo_texture.sampler)
                .build();

            let metallic_roughness_texture = mesh
                .metallic_roughness_texture
                .as_ref()
                .unwrap_or(&default_metallic_roughness);
            let metallic_roughness_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(metallic_roughness_texture.image_view)
                .sampler(metallic_roughness_texture.sampler)
                .build();

            let normal_texture = mesh.normal_texture.as_ref().unwrap_or(&default_normal);
            let normal_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(normal_texture.image_view)
                .sampler(normal_texture.sampler)
                .build();

            let descriptor_writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_info))
                    .build(),
                // albedo
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&albedo_info))
                    .build(),
                // metallic
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&metallic_roughness_info))
                    .build(),
                // norm
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&normal_info))
                    .build(),
            ];

            unsafe {
                device.update_descriptor_sets(&descriptor_writes, &[]);
            }
        }

        descriptor_sets
    }

    fn create_render_pass(device: &Device, surface_format: vk::SurfaceFormatKHR) -> vk::RenderPass {
        let attachments = [
            vk::AttachmentDescription::builder()
                .format(surface_format.format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build(),
            vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .build(),
        ];
        let color_reference = [vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];
        let depth_reference = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let subpasses = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_reference)
            .depth_stencil_attachment(&depth_reference)
            .build()];
        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses);
        unsafe {
            device
                .create_render_pass(&render_pass_create_info, None)
                .expect("Failed to create render pass")
        }
    }

    fn create_framebuffers(
        device: &Device,
        allocator: Arc<Mutex<Allocator>>,
        render_pass: vk::RenderPass,
        format: vk::SurfaceFormatKHR,
        extent: vk::Extent2D,
        swapchain_images: &[vk::Image],
    ) -> CreateFramebuffersResult {
        let mut framebuffers = vec![];
        let mut depth_images_and_allocations = vec![];
        let mut color_image_views = vec![];
        let mut depth_image_views = vec![];
        for &image in swapchain_images.iter() {
            let mut attachments = vec![];

            let color_attachment = unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            ),
                        None,
                    )
                    .expect("Failed to create image view")
            };
            attachments.push(color_attachment);
            color_image_views.push(color_attachment);
            let depth_image_create_info = vk::ImageCreateInfo::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });
            let depth_image = unsafe {
                device
                    .create_image(&depth_image_create_info, None)
                    .expect("Failed to create image")
            };
            let depth_allocation = allocator
                .lock()
                .unwrap()
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "Depth Image",
                    requirements: unsafe { device.get_image_memory_requirements(depth_image) },
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .expect("Failed to allocate memory");
            unsafe {
                device
                    .bind_image_memory(
                        depth_image,
                        depth_allocation.memory(),
                        depth_allocation.offset(),
                    )
                    .expect("Failed to bind image memory")
            };
            let depth_attachment = unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(depth_image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::D32_SFLOAT)
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            ),
                        None,
                    )
                    .expect("Failed to create depth image view")
            };
            attachments.push(depth_attachment);
            depth_image_views.push(depth_attachment);
            framebuffers.push(unsafe {
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::builder()
                            .render_pass(render_pass)
                            .attachments(attachments.as_slice())
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer")
            });
            depth_images_and_allocations.push((depth_image, depth_allocation));
        }
        CreateFramebuffersResult(
            framebuffers,
            depth_images_and_allocations,
            color_image_views,
            depth_image_views,
        )
    }

    fn create_graphics_pipeline(
        device: &Device,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        render_pass: vk::RenderPass,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vertex_shader_module = {
            let spirv = include_spirv!("../../../shader-cache/main.vert.spv");
            let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&spirv);
            unsafe {
                device
                    .create_shader_module(&shader_module_create_info, None)
                    .expect("Failed to create shader module")
            }
        };
        let fragment_shader_module = {
            let spirv = include_spirv!("../../../shader-cache/main.frag.spv");
            let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&spirv);
            unsafe {
                device
                    .create_shader_module(&shader_module_create_info, None)
                    .expect("Failed to create shader module")
            }
        };
        let main_function_name_fs = CString::new("main").unwrap();
        let main_function_name_vs = CString::new("main").unwrap();
        let pipeline_shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader_module)
                .name(&main_function_name_vs)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader_module)
                .name(&main_function_name_fs)
                .build(),
        ];

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .build();

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(descriptor_set_layouts)
                        .push_constant_ranges(&[push_constant_range]),
                    None,
                )
                .expect("Failed to create pipeline layout")
        };
        let vertex_input_binding = Vertex::get_binding_descriptions();
        let vertex_input_attribute = Vertex::get_attribute_descriptions();
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);
        let stencil_op = vk::StencilOpState::builder()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS);
        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .front(*stencil_op)
            .back(*stencil_op);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            );
        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(std::slice::from_ref(&color_blend_attachment));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_input_attribute)
            .vertex_binding_descriptions(&vertex_input_binding);
        let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&pipeline_shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blend_info)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);
        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_create_info),
                    None,
                )
                .unwrap()[0]
        };
        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }

        (graphics_pipeline, pipeline_layout)
    }

    fn create_command_buffers(
        device: &Device,
        command_pool: vk::CommandPool,
        swapchain_count: usize,
    ) -> Vec<vk::CommandBuffer> {
        unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(swapchain_count as u32),
                )
                .expect("Failed to allocate command buffers")
        }
    }

    fn create_sync_objects(
        device: &Device,
        swapchain_count: usize,
    ) -> (Vec<vk::Fence>, Vec<vk::Semaphore>, Vec<vk::Semaphore>) {
        let fence_create_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED); // Start signaled

        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();

        let mut in_flight_fences = Vec::with_capacity(swapchain_count);
        let mut image_available_semaphores = Vec::with_capacity(swapchain_count);
        let mut render_finished_semaphores = Vec::with_capacity(swapchain_count);

        for _ in 0..swapchain_count {
            unsafe {
                let fence = device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create fence");
                let image_available = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create semaphore");
                let render_finished = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create semaphore");

                in_flight_fences.push(fence);
                image_available_semaphores.push(image_available);
                render_finished_semaphores.push(render_finished);
            }
        }

        (
            in_flight_fences,
            image_available_semaphores,
            render_finished_semaphores,
        )
    }

    fn recreate_swapchain(&mut self, width: u32, height: u32, egui_cmd: &mut EguiCommand) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle")
        };

        unsafe {
            let mut allocator = self.allocator.lock().unwrap();
            for &fence in self.in_flight_fences.iter() {
                self.device.destroy_fence(fence, None);
            }
            for &semaphore in self.image_available_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in self.render_finished_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &framebuffer in self.framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for &image_view in self.color_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            for &image_view in self.depth_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            for (image, allocation) in self.depth_images_and_allocations.drain(..) {
                self.device.destroy_image(image, None);
                allocator.free(allocation).expect("Failed to free memory");
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        self.width = width;
        self.height = height;

        let (swapchain, swapchain_images, surface_format, surface_extent) = {
            let surface_capabilities = unsafe {
                self.surface_loader
                    .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                    .expect("Failed to get physical device surface capabilities")
            };
            let surface_formats = unsafe {
                self.surface_loader
                    .get_physical_device_surface_formats(self.physical_device, self.surface)
                    .expect("Failed to get physical device surface formats")
            };

            let surface_format = *surface_formats
                .iter()
                .find(|surface_format| {
                    surface_format.format == vk::Format::B8G8R8A8_UNORM
                        || surface_format.format == vk::Format::R8G8B8A8_UNORM
                })
                .unwrap_or(&surface_formats[0]);

            let surface_present_mode = vk::PresentModeKHR::FIFO;

            let surface_extent = if surface_capabilities.current_extent.width != u32::MAX {
                surface_capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: self.width.clamp(
                        surface_capabilities.min_image_extent.width,
                        surface_capabilities.max_image_extent.width,
                    ),
                    height: self.height.clamp(
                        surface_capabilities.min_image_extent.height,
                        surface_capabilities.max_image_extent.height,
                    ),
                }
            };

            let image_count = surface_capabilities.min_image_count + 1;
            let image_count = if surface_capabilities.max_image_count != 0 {
                image_count.min(surface_capabilities.max_image_count)
            } else {
                image_count
            };

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(self.surface)
                .min_image_count(image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_extent)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(surface_present_mode)
                .image_array_layers(1)
                .clipped(true);
            let swapchain = unsafe {
                self.swapchain_loader
                    .create_swapchain(&swapchain_create_info, None)
                    .expect("Failed to create swapchain")
            };

            let swapchain_images = unsafe {
                self.swapchain_loader
                    .get_swapchain_images(swapchain)
                    .expect("Failed to get swapchain images")
            };

            (swapchain, swapchain_images, surface_format, surface_extent)
        };
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.surface_format = surface_format;
        self.surface_extent = surface_extent;

        egui_cmd.update_swapchain(egui_ash::SwapchainUpdateInfo {
            swapchain_images: self.swapchain_images.clone(),
            surface_format: self.surface_format.format,
            width: self.width,
            height: self.height,
        });

        self.render_pass = Self::create_render_pass(&self.device, self.surface_format);

        let CreateFramebuffersResult(
            framebuffers,
            depth_images_and_allocations,
            color_image_views,
            depth_image_views,
        ) = Self::create_framebuffers(
            &self.device,
            Arc::clone(&self.allocator),
            self.render_pass,
            self.surface_format,
            self.surface_extent,
            &self.swapchain_images,
        );
        self.framebuffers = framebuffers;
        self.depth_images_and_allocations = depth_images_and_allocations;
        self.color_image_views = color_image_views;
        self.depth_image_views = depth_image_views;

        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            Self::create_sync_objects(&self.device, self.swapchain_images.len());
        self.in_flight_fences = in_flight_fences;
        self.image_available_semaphores = image_available_semaphores;
        self.render_finished_semaphores = render_finished_semaphores;

        self.current_frame = 0;
        self.dirty_swapchain = false;
    }

    fn new(
        physical_device: vk::PhysicalDevice,
        device: Device,
        surface_loader: Surface,
        swapchain_loader: Swapchain,
        allocator: Arc<Mutex<Allocator>>,
        surface: vk::SurfaceKHR,
        queue_family_index: u32,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
    ) -> Self {
        let (swapchain, surface_format, surface_extent, swapchain_images) = Self::create_swapchain(
            physical_device,
            &surface_loader,
            &swapchain_loader,
            surface,
            queue_family_index,
            width,
            height,
        );
        let (uniform_buffers, uniform_buffer_allocations) =
            Self::create_uniform_buffers::<UniformBufferObject>(
                &device,
                allocator.clone(),
                swapchain_images.len(),
            );

        let t = Instant::now();
        tracing::info!("loading model!");
        let mut model = Model::load(
            &device,
            allocator.clone(),
            command_pool,
            queue,
            "./sponza/NewSponza_Main_glTF_003.gltf",
        );
        tracing::info!(
            "loaded {} meshes in {:.2} seconds!",
            model.meshes.len(),
            t.elapsed().as_secs_f32()
        );

        let descriptor_pool =
            Self::create_descriptor_pool(&device, model.meshes.len() * swapchain_images.len());
        let descriptor_set_layouts =
            Self::create_descriptor_set_layouts(&device, swapchain_images.len());

        let render_pass = Self::create_render_pass(&device, surface_format);
        let CreateFramebuffersResult(
            framebuffers,
            depth_images_and_allocations,
            color_image_views,
            depth_image_views,
        ) = Self::create_framebuffers(
            &device,
            allocator.clone(),
            render_pass,
            surface_format,
            surface_extent,
            &swapchain_images,
        );
        let (pipeline, pipeline_layout) =
            Self::create_graphics_pipeline(&device, &descriptor_set_layouts, render_pass);
        let command_buffers =
            Self::create_command_buffers(&device, command_pool, swapchain_images.len());
        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            Self::create_sync_objects(&device, swapchain_images.len());

        for mesh in &mut model.meshes {
            mesh.descriptor_sets = Self::create_descriptor_sets(
                &device,
                descriptor_pool,
                &descriptor_set_layouts,
                &uniform_buffers,
                allocator.clone(),
                command_pool,
                queue,
                &mesh,
            );
        }

        Self {
            width,
            height,
            physical_device,
            device,
            surface_loader,
            swapchain_loader,
            allocator: ManuallyDrop::new(allocator),
            surface,
            queue,
            swapchain,
            surface_format,
            surface_extent,
            swapchain_images,
            uniform_buffers,
            uniform_buffer_allocations,
            descriptor_pool,
            descriptor_set_layouts,
            render_pass,
            framebuffers,
            depth_images_and_allocations,
            color_image_views,
            depth_image_views,
            pipeline,
            pipeline_layout,
            model,
            command_buffers,
            in_flight_fences,
            image_available_semaphores,
            render_finished_semaphores,
            current_frame: 0,
            dirty_swapchain: true,

            frame_counter: 0,
            camera_position: Vec3::new(0.0, 0.0, -5.0),
            camera_yaw: 0.,
            camera_pitch: 0.,
            camera_fov: 90.,
            bg_color: Vec3::splat(0.1),
            model_color: Vec3::splat(0.8),
            accumulation_reset_needed: true,
        }
    }

    pub fn update_colors(&mut self, bg_color: Vec3, model_color: Vec3) {
        let bg_color = clamped_color(bg_color);
        let model_color = clamped_color(model_color);

        if bg_color != self.bg_color {
            self.bg_color = bg_color;
            self.accumulation_reset_needed = true;
        }

        if model_color != self.model_color {
            self.model_color = model_color;
            self.accumulation_reset_needed = true;
        }
    }

    pub fn update_camera(&mut self, new_position: Vec3, yaw: f32, pitch: f32, fov: f32) {
        if (new_position - self.camera_position).length() > 0.0001
            || (yaw - self.camera_yaw).abs() > 0.001
            || (pitch - self.camera_pitch).abs() > 0.001
            || (fov - self.camera_fov).abs() > 0.001
        {
            self.camera_position = new_position;
            self.camera_yaw = yaw;
            self.camera_pitch = pitch;
            self.camera_fov = fov;
            self.accumulation_reset_needed = true;
        }
    }

    pub fn render(&mut self, width: u32, height: u32, mut egui_cmd: EguiCommand, rotate_y: f32) {
        puffin::profile_function!();
        if width == 0 || height == 0 {
            return;
        }

        if self.dirty_swapchain || self.accumulation_reset_needed {
            self.frame_counter = 0;
            self.accumulation_reset_needed = false;
        }

        if self.dirty_swapchain
            || width != self.width
            || height != self.height
            || egui_cmd.swapchain_recreate_required()
        {
            puffin::profile_scope!("recreate_swapchain");
            self.recreate_swapchain(width, height, &mut egui_cmd);
        }

        unsafe {
            puffin::profile_scope!("wait_for_fences");
            self.device.wait_for_fences(
                std::slice::from_ref(&self.in_flight_fences[self.current_frame]),
                true,
                u64::MAX,
            )
        }
        .expect("Failed to wait for fences");

        unsafe {
            self.device.reset_fences(std::slice::from_ref(
                &self.in_flight_fences[self.current_frame],
            ))
        }
        .expect("Failed to reset fences");

        let result = unsafe {
            puffin::profile_scope!("acquire_next_image");
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            )
        };
        let index = match result {
            Ok((index, _)) => index as usize,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.dirty_swapchain = true;
                return;
            }
            Err(_) => return,
        };

        let view = {
            puffin::profile_scope!("calculate_view");
            let (sin_pitch, cos_pitch) = self.camera_pitch.sin_cos();
            let (sin_yaw, cos_yaw) = self.camera_yaw.sin_cos();

            let look_dir =
                Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw).normalize();

            Mat4::look_at_rh(
                self.camera_position,
                self.camera_position + look_dir,
                Vec3::new(0.0, -1.0, 0.0),
            )
        };

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            puffin::profile_scope!("render");
            self.device
                .begin_command_buffer(
                    self.command_buffers[self.current_frame],
                    &command_buffer_begin_info,
                )
                .expect("Failed to begin command buffer");

            self.device.cmd_begin_render_pass(
                self.command_buffers[self.current_frame],
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[index])
                    .render_area(
                        vk::Rect2D::builder()
                            .offset(vk::Offset2D::builder().x(0).y(0).build())
                            .extent(self.surface_extent)
                            .build(),
                    )
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [self.bg_color.x, self.bg_color.y, self.bg_color.z, 1.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ]),
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                self.command_buffers[self.current_frame],
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            self.device.cmd_set_viewport(
                self.command_buffers[self.current_frame],
                0,
                std::slice::from_ref(
                    &vk::Viewport::builder()
                        .width(width as f32)
                        .height(height as f32)
                        .min_depth(0.0)
                        .max_depth(1.0),
                ),
            );
            self.device.cmd_set_scissor(
                self.command_buffers[self.current_frame],
                0,
                std::slice::from_ref(
                    &vk::Rect2D::builder()
                        .offset(vk::Offset2D::builder().build())
                        .extent(self.surface_extent),
                ),
            );

            for mesh in &self.model.meshes {
                let push_constants = PushConstants {
                    texture_size: Vec4::new(1024.0, 1024.0, 1.0 / 1024.0, 1.0 / 1024.0),
                };
                unsafe {
                    self.device.cmd_push_constants(
                        self.command_buffers[self.current_frame],
                        self.pipeline_layout,
                        vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::cast_slice(&[push_constants]),
                    );
                }
                let model_matrix = Mat4::from_rotation_y(rotate_y.to_radians()) * mesh.transform;

                let ubo = UniformBufferObject {
                    model: model_matrix,
                    view,
                    proj: Mat4::perspective_rh(
                        self.camera_fov.to_radians(),
                        width as f32 / height as f32,
                        0.1,
                        1000.0,
                    ),
                    camera_pos: self.camera_position,
                    material: Material {
                        base_color: mesh.base_color,
                        metallic_factor: mesh.metallic_factor,
                        roughness_factor: mesh.roughness_factor,
                        _padding: [0.0, 0.0],
                    },
                };

                let ptr = self.uniform_buffer_allocations[self.current_frame]
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr() as *mut UniformBufferObject;
                ptr.copy_from_nonoverlapping(&ubo, 1);

                self.device.cmd_bind_descriptor_sets(
                    self.command_buffers[self.current_frame],
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[mesh.descriptor_sets[self.current_frame]],
                    &[],
                );

                self.device.cmd_bind_vertex_buffers(
                    self.command_buffers[self.current_frame],
                    0,
                    &[mesh.vertex_buffer],
                    &[0],
                );
                self.device.cmd_draw(
                    self.command_buffers[self.current_frame],
                    mesh.vertex_count,
                    1,
                    0,
                    0,
                );
            }

            self.device
                .cmd_end_render_pass(self.command_buffers[self.current_frame]);

            egui_cmd.record(self.command_buffers[self.current_frame], index);

            self.device
                .end_command_buffer(self.command_buffers[self.current_frame])
                .expect("Failed to end command buffer");
        }

        let buffers_to_submit = [self.command_buffers[self.current_frame]];
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&buffers_to_submit)
            .wait_semaphores(std::slice::from_ref(
                &self.image_available_semaphores[self.current_frame],
            ))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .signal_semaphores(std::slice::from_ref(
                &self.render_finished_semaphores[self.current_frame],
            ));
        unsafe {
            puffin::profile_scope!("queue_submit");
            self.device
                .queue_submit(
                    self.queue,
                    std::slice::from_ref(&submit_info),
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to submit queue");
        };

        let image_indices = [index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(std::slice::from_ref(
                &self.render_finished_semaphores[self.current_frame],
            ))
            .swapchains(std::slice::from_ref(&self.swapchain))
            .image_indices(&image_indices);
        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        };
        let is_dirty_swapchain = match result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => true,
            Err(error) => panic!("Failed to present queue. Cause: {}", error),
            _ => false,
        };
        self.dirty_swapchain = is_dirty_swapchain;

        self.current_frame = (self.current_frame + 1) % self.in_flight_fences.len();

        self.frame_counter += 1;

        static mut LAST_ROTATE_Y: f32 = 0.0;
        unsafe {
            if (LAST_ROTATE_Y - rotate_y).abs() > 0.001 {
                self.accumulation_reset_needed = true;
                LAST_ROTATE_Y = rotate_y;
            }
        }
    }

    fn destroy(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");

            let mut allocator = self.allocator.lock().unwrap();
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence, None);
            }
            for semaphore in self.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            self.model.destroy(&self.device, &mut allocator);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            for &framebuffer in self.framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for &color_image_view in self.color_image_views.iter() {
                self.device.destroy_image_view(color_image_view, None);
            }
            for &depth_image_view in self.depth_image_views.iter() {
                self.device.destroy_image_view(depth_image_view, None);
            }
            for (depth_image, allocation) in self.depth_images_and_allocations.drain(..) {
                self.device.destroy_image(depth_image, None);
                allocator.free(allocation).expect("Failed to free memory");
            }
            self.device.destroy_render_pass(self.render_pass, None);
            for &descriptor_set_layout in self.descriptor_set_layouts.iter() {
                self.device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for &uniform_buffer in self.uniform_buffers.iter() {
                self.device.destroy_buffer(uniform_buffer, None);
            }
            for allocation in self.uniform_buffer_allocations.drain(..) {
                allocator.free(allocation).expect("Failed to free memory");
            }

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
        }
    }
}

#[derive(Clone)]
pub struct Renderer {
    pub inner: Arc<Mutex<RendererInner>>,
}
impl Renderer {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        device: Device,
        surface_loader: Surface,
        swapchain_loader: Swapchain,
        allocator: Arc<Mutex<Allocator>>,
        surface: vk::SurfaceKHR,
        queue_family_index: u32,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RendererInner::new(
                physical_device,
                device,
                surface_loader,
                swapchain_loader,
                allocator,
                surface,
                queue_family_index,
                queue,
                command_pool,
                width,
                height,
            ))),
        }
    }

    pub fn destroy(&mut self) {
        self.inner.lock().unwrap().destroy();
    }
}

fn clamped_color(color: Vec3) -> Vec3 {
    Vec3::new(
        color.x.clamp(0.0, 1.0),
        color.y.clamp(0.0, 1.0),
        color.z.clamp(0.0, 1.0),
    )
}

fn create_default_texture(
    device: &Device,
    allocator: Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Arc<Texture> {
    let white_pixel = vec![255u8, 255, 255, 255];
    Arc::new(Texture::new(
        device,
        allocator,
        command_pool,
        queue,
        1,
        1,
        &white_pixel,
    ))
}

fn create_default_metallic_roughness_texture(
    device: &Device,
    allocator: Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Arc<Texture> {
    let pixel = vec![0u8, 0, 255, 255]; // Non-metallic (0), rough (1.0)
    Arc::new(Texture::new(
        device,
        allocator,
        command_pool,
        queue,
        1,
        1,
        &pixel,
    ))
}

fn create_default_normal_texture(
    device: &Device,
    allocator: Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Arc<Texture> {
    let pixel = vec![128u8, 128, 255, 255]; // Default normal pointing up
    Arc::new(Texture::new(
        device,
        allocator,
        command_pool,
        queue,
        1,
        1,
        &pixel,
    ))
}
