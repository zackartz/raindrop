mod error;
mod geo;
mod texture;

use std::{
    collections::HashMap,
    fs,
    hash::Hash,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

use ash::vk;
use gfx_hal::{device::Device, instance::Instance, queue::Queue, Fence};
use tracing::{debug, error, trace, warn};

pub use error::{ResourceManagerError, Result};
pub use geo::Geometry;
pub use texture::{Material, SamplerDesc, Texture};

use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
    MemoryLocation,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageHandle(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] // New Handle
pub struct ImageViewHandle(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerHandle(u64);

#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub handle: BufferHandle,
    pub buffer: vk::Buffer,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub mapped_ptr: Option<*mut u8>,
}

#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub handle: ImageHandle,
    pub image: vk::Image,
    // pub view: vk::ImageView, // Removed raw view
    pub default_view_handle: Option<ImageViewHandle>, // Added handle to default view
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub usage: vk::ImageUsageFlags,
    pub layout: vk::ImageLayout,
    pub mapped_ptr: Option<*mut u8>,
}

#[derive(Debug, Clone)]
pub struct SamplerInfo {
    pub handle: SamplerHandle,
    pub sampler: vk::Sampler,
    pub desc: SamplerDesc, // Include desc if useful
}

struct InternalBufferInfo {
    device: Arc<Device>,              // Keep device alive for Drop
    allocator: Arc<Mutex<Allocator>>, // Needed for Drop
    buffer: vk::Buffer,
    allocation: Option<Allocation>, // Option because it's taken in Drop
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    handle: BufferHandle,
}

impl Drop for InternalBufferInfo {
    fn drop(&mut self) {
        trace!("Dropping InternalBufferInfo for handle: {:?}", self.handle);
        if let Some(allocation) = self.allocation.take() {
            let mut allc = self.allocator.lock().expect("to acquire mutex lock");
            if let Err(e) = allc.free(allocation) {
                error!(
                    "Failed to free allocation for buffer handle {:?}, {}",
                    self.handle, e
                );
            } else {
                trace!("Freed alloation for buffer handle: {:?}", self.handle);
            }
        }
        unsafe {
            self.device.raw().destroy_buffer(self.buffer, None);
        }
        trace!("Destroyed vk::Buffer for handle {:?}", self.handle);
    }
}

struct InternalImageInfo {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    image: vk::Image,
    // view: vk::ImageView, // Removed raw view
    default_view_handle: Option<ImageViewHandle>, // Added handle
    allocation: Option<Allocation>,
    format: vk::Format,
    extent: vk::Extent3D,
    usage: vk::ImageUsageFlags,
    layout: vk::ImageLayout,
    handle: ImageHandle,
}

impl Drop for InternalImageInfo {
    fn drop(&mut self) {
        trace!("Dropping InternalImageInfo for handle {:?}", self.handle);

        // Then free memory
        if let Some(allocation) = self.allocation.take() {
            let mut allocator = self.allocator.lock().expect("to acquire mutex lock");
            if let Err(e) = allocator.free(allocation) {
                error!(
                    "Failed to free allocation for image handle {:?}: {}",
                    self.handle, e
                );
            } else {
                trace!("Freed allocation for image handle {:?}", self.handle);
            }
        }
        // Then destroy image
        unsafe {
            self.device.raw().destroy_image(self.image, None);
        }
        trace!(
            "Destroyed vk::Image/vk::ImageView for handle {:?}",
            self.handle
        );
    }
}

struct InternalSamplerInfo {
    device: Arc<Device>,
    sampler: vk::Sampler,
    handle: SamplerHandle,
    desc: SamplerDesc,
}

struct InternalImageViewInfo {
    device: Arc<Device>, // Keep device alive for Drop
    view: vk::ImageView,
    handle: ImageViewHandle,
    // Optional: Store the ImageHandle this view belongs to for debugging/validation
    // image_handle: ImageHandle,
}

#[derive(Debug, Clone)]
pub struct ImageViewInfo {
    pub handle: ImageViewHandle,
    pub view: vk::ImageView,
    // Could add format, subresource range etc. if needed frequently
}

impl Drop for InternalImageViewInfo {
    fn drop(&mut self) {
        trace!(
            "Dropping InternalImageViewInfo for handle {:?}",
            self.handle
        );
        unsafe {
            self.device.raw().destroy_image_view(self.view, None);
        }
        trace!("Destroyed vk::ImageView for handle {:?}", self.handle);
    }
}

impl Drop for InternalSamplerInfo {
    fn drop(&mut self) {
        trace!("Dropping InternalSamplerInfo for handle {:?}", self.handle);
        unsafe {
            self.device.raw().destroy_sampler(self.sampler, None);
        }
        trace!("Destroyed vk::Sampler for handle {:?}", self.handle);
    }
}

struct TransferSetup {
    command_pool: vk::CommandPool,
    queue: Arc<Queue>,
    fence: Fence,
}

pub struct ResourceManager {
    _instance: Arc<Instance>,
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    buffers: Arc<Mutex<HashMap<u64, InternalBufferInfo>>>,
    images: Arc<Mutex<HashMap<u64, InternalImageInfo>>>,
    image_views: Arc<Mutex<HashMap<u64, InternalImageViewInfo>>>,
    samplers: Arc<Mutex<HashMap<u64, InternalSamplerInfo>>>,
    sampler_cache_by_desc: Arc<Mutex<HashMap<SamplerDesc, SamplerHandle>>>,
    texture_cache_uri: Arc<Mutex<HashMap<PathBuf, Arc<Texture>>>>,
    next_id: AtomicU64,
    transfer_setup: Arc<Mutex<TransferSetup>>,
}

impl ResourceManager {
    /// Creates a new ResourceManager.
    pub fn new(instance: Arc<Instance>, device: Arc<Device>) -> Result<Self> {
        debug!("Initializing ResourceManager...");
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.ash_instance().clone(),
            device: device.raw().clone(),
            physical_device: device.physical_device_handle(),
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;
        debug!("GPU Allocator created.");

        let queue_family_index = device
            .transfer_queue_family_index()
            .or(device.compute_queue_family_index()) // Try compute as fallback
            .unwrap_or(device.graphics_queue_family_index()); // Graphics as last resort

        let queue = device.get_queue(queue_family_index, 0)?;

        // Create command pool for transfer commands
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT) // Hint that buffers are short-lived
            .queue_family_index(queue_family_index);
        let command_pool = unsafe { device.raw().create_command_pool(&pool_info, None)? };

        // Create a fence for waiting
        let fence = Fence::new(device.clone(), false)?;

        let new_setup = TransferSetup {
            command_pool,
            queue,
            fence,
        };

        Ok(Self {
            _instance: instance,
            device,
            allocator: Arc::new(Mutex::new(allocator)),
            buffers: Arc::new(Mutex::new(HashMap::new())),
            images: Arc::new(Mutex::new(HashMap::new())),
            image_views: Arc::new(Mutex::new(HashMap::new())), // Initialize view map
            samplers: Arc::new(Mutex::new(HashMap::new())),
            sampler_cache_by_desc: Arc::new(Mutex::new(HashMap::new())),
            texture_cache_uri: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            transfer_setup: Arc::new(Mutex::new(new_setup)),
        })
    }

    /// Gets a shared reference to the Allocator
    pub fn allocator(&self) -> Arc<Mutex<Allocator>> {
        self.allocator.clone()
    }

    /// Helper to allocate, begin, end, submit, and wait for a single command buffer
    /// using the provided TransferSetup.
    unsafe fn submit_commands_and_wait<F>(
        &self,
        transfer_setup: &TransferSetup, // Use the cloned setup
        record_fn: F,
    ) -> Result<()>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<()>, // Closure records commands
    {
        let device_raw = self.device.raw(); // Get raw ash::Device

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(transfer_setup.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = device_raw.allocate_command_buffers(&alloc_info)?[0];
        tracing::info!("Allocated command_buffer: {:?}", command_buffer);
        trace!("Allocated temporary command buffer for transfer.");

        // Begin recording
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device_raw.begin_command_buffer(command_buffer, &begin_info)?;

        // --- Record user commands ---
        let record_result = record_fn(command_buffer);
        // --- End Recording ---
        // Always end buffer, even if recording failed, to allow cleanup
        device_raw.end_command_buffer(command_buffer)?;

        // Check user function result *after* ending buffer
        record_result?;
        trace!("Transfer commands recorded.");

        // Submit to the transfer queue
        let submits =
            [vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer))];
        // Use the queue from the TransferSetup. Assuming Queue::submit handles locking.
        transfer_setup
            .queue
            .submit(device_raw, &submits, Some(&transfer_setup.fence))?; // Submit WITH fence
        trace!("Transfer command buffer submitted.");

        // Wait for completion using the fence
        transfer_setup.fence.wait(None)?;

        // Free command buffer *after* successful wait
        device_raw.free_command_buffers(transfer_setup.command_pool, &[command_buffer]);
        trace!("Temporary command buffer freed.");

        transfer_setup.fence.reset()?;

        Ok(())
    }

    /// Creates a Vulkan buffer and allocates/binds memory for it.
    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<BufferHandle> {
        trace!(
            "Creating buffer: size={}, usage={:?}, location={:?}",
            size,
            usage,
            location
        );
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE); // Assuming exclusive access

        let buffer = unsafe { self.device.raw().create_buffer(&buffer_info, None)? };

        let requirements = unsafe { self.device.raw().get_buffer_memory_requirements(buffer) };

        let allocation = self.allocator.lock()?.allocate(&AllocationCreateDesc {
            name: &format!("buffer_usage_{:?}_loc_{:?}", usage, location),
            requirements,
            location,
            linear: true, // Buffers are linear
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            self.device.raw().bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )?;
        }
        trace!("Buffer memory bound.");

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let handle = BufferHandle(id);

        let internal_info = InternalBufferInfo {
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            buffer,
            allocation: Some(allocation),
            size,
            usage,
            handle,
        };

        self.buffers.lock()?.insert(id, internal_info);
        debug!("Buffer created successfully: handle={:?}", handle);
        Ok(handle)
    }

    // Make sure create_buffer_init is correct and doesn't call itself
    pub fn create_buffer_init(
        &self,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        data: &[u8],
    ) -> Result<BufferHandle> {
        let size = data.len() as vk::DeviceSize;
        if size == 0 {
            return Err(ResourceManagerError::Other(
                "Cannot create buffer with empty data".to_string(),
            ));
        }
        debug!(
            "Creating buffer with init data: size={}, usage={:?}, location={:?}",
            size, usage, location
        );

        // 1. Create Staging Buffer (CPU accessible)
        let staging_handle = self.create_buffer(
            // Call create_buffer, NOT create_buffer_init
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;

        // 2. Map & Copy data to staging buffer
        {
            let staging_info = self.get_buffer_info(staging_handle)?;
            let mapping = staging_info
                .mapped_ptr
                .ok_or(ResourceManagerError::MappingFailed)?;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), mapping, data.len());
            }
            // Optional: Flush
            trace!("Data copied to staging buffer handle {:?}", staging_handle);
        }

        // 3. Create Destination Buffer
        let final_usage = usage | vk::BufferUsageFlags::TRANSFER_DST;
        let dest_handle = self.create_buffer(size, final_usage, location)?; // Call create_buffer

        // 4. Perform Copy via Command Buffer
        {
            let transfer_setup_locked = self.transfer_setup.lock()?;
            let buffers_locked = self.buffers.lock()?;

            let dest_internal = buffers_locked
                .get(&dest_handle.0)
                .ok_or(ResourceManagerError::HandleNotFound(dest_handle.0))?;
            let staging_internal = buffers_locked
                .get(&staging_handle.0)
                .ok_or(ResourceManagerError::HandleNotFound(staging_handle.0))?;

            trace!("Submitting buffer copy command...");
            unsafe {
                Self::submit_commands_and_wait(self, &transfer_setup_locked, |cmd| {
                    let region = vk::BufferCopy::default().size(size);
                    self.device.raw().cmd_copy_buffer(
                        cmd,
                        staging_internal.buffer,
                        dest_internal.buffer,
                        &[region],
                    );
                    Ok(())
                })?;
            }
            trace!("Buffer copy command finished.");
        }

        // 5. Cleanup staging buffer
        self.destroy_buffer(staging_handle)?;
        debug!("Staging buffer destroyed: handle={:?}", staging_handle);

        Ok(dest_handle)
    }

    /// Creates a Vulkan image and allocates/binds memory for it.
    /// Also creates a default `ImageView`.
    /// Does not handle data uploads or layout transitions.
    pub fn create_image(
        &self,
        create_info: &vk::ImageCreateInfo, // User provides image details
        location: MemoryLocation,
        aspect_flags: vk::ImageAspectFlags,
    ) -> Result<ImageHandle> {
        trace!(
            "Creating image: format={:?}, extent={:?}, usage={:?}, location={:?}",
            create_info.format,
            create_info.extent,
            create_info.usage,
            location
        );

        let image = unsafe { self.device.raw().create_image(create_info, None)? };
        let requirements = unsafe { self.device.raw().get_image_memory_requirements(image) };
        let allocation = self.allocator.lock()?.allocate(&AllocationCreateDesc {
            name: &format!(
                "image_fmt_{:?}_usage_{:?}",
                create_info.format, create_info.usage
            ),
            requirements,
            location,
            linear: create_info.tiling == vk::ImageTiling::LINEAR,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            self.device
                .raw()
                .bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }
        trace!("Image memory bound.");

        // 2. Create the *default* Image View using the new method
        let default_view_create_info = Self::build_default_view_info(create_info, aspect_flags);
        // Use internal helper to avoid locking images map again if called from create_image_view
        let default_view_handle =
            self.create_image_view_internal(image, &default_view_create_info)?;
        trace!(
            "Default image view created: handle={:?}",
            default_view_handle
        );

        // 3. Store InternalImageInfo
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let handle = ImageHandle(id);

        let internal_info = InternalImageInfo {
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            image,
            default_view_handle: Some(default_view_handle), // Store handle
            allocation: Some(allocation),
            format: create_info.format,
            extent: create_info.extent,
            usage: create_info.usage,
            layout: create_info.initial_layout,
            handle,
        };

        self.images.lock()?.insert(id, internal_info);
        debug!("Image created successfully: handle={:?}", handle);
        Ok(handle)
    }

    fn build_default_view_info<'a>(
        image_create_info: &vk::ImageCreateInfo,
        aspect_flags: vk::ImageAspectFlags,
    ) -> vk::ImageViewCreateInfo<'a> {
        let view_type = match image_create_info.image_type {
            vk::ImageType::TYPE_1D => {
                if image_create_info.array_layers > 1 {
                    vk::ImageViewType::TYPE_1D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_1D
                }
            }
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => {
                // TYPE_2D
                if image_create_info
                    .flags
                    .contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                {
                    if image_create_info.array_layers > 6 {
                        vk::ImageViewType::CUBE_ARRAY
                    } else {
                        vk::ImageViewType::CUBE
                    } // Assumes 6 layers
                } else if image_create_info.array_layers > 1 {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                }
            }
        };

        vk::ImageViewCreateInfo::default()
            // .image(image) // Image is set by create_image_view_internal
            .view_type(view_type)
            .format(image_create_info.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_flags)
                    .base_mip_level(0)
                    .level_count(image_create_info.mip_levels)
                    .base_array_layer(0)
                    .layer_count(image_create_info.array_layers),
            )
        // .components(...) // Default components usually fine
    }

    /// Creates a new Vulkan ImageView for an existing Image.
    /// The view's lifetime is managed by the ResourceManager.
    pub fn create_image_view(
        &self,
        image_handle: ImageHandle,
        view_create_info: &vk::ImageViewCreateInfo, // User provides desired view settings
    ) -> Result<ImageViewHandle> {
        trace!("Request to create image view for image {:?}", image_handle);
        // 1. Get the vk::Image handle from the InternalImageInfo
        let image_vk_handle = {
            // Scope for images lock
            let images_map = self.images.lock()?;
            let internal_image_info = images_map
                .get(&image_handle.0)
                .ok_or(ResourceManagerError::HandleNotFound(image_handle.0))?;
            internal_image_info.image // Copy the vk::Image handle
        }; // Release images lock

        // 2. Call internal helper to create the view and manage it
        self.create_image_view_internal(image_vk_handle, view_create_info)
    }

    /// Internal helper to create and register an image view.
    /// Takes the raw vk::Image to avoid re-locking the images map.
    fn create_image_view_internal(
        &self,
        image: vk::Image, // The actual Vulkan image handle
        view_create_info: &vk::ImageViewCreateInfo,
    ) -> Result<ImageViewHandle> {
        // Ensure the create info points to the correct image
        let final_view_info = (*view_create_info).image(image);

        let view = unsafe {
            self.device
                .raw()
                .create_image_view(&final_view_info, None)?
        };
        trace!("vk::ImageView created.");

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let handle = ImageViewHandle(id);

        let internal_view_info = InternalImageViewInfo {
            device: self.device.clone(),
            view,
            handle,
            // image_handle: image_handle, // Optional: Store originating image handle
        };

        // Add to the image_views map
        self.image_views.lock()?.insert(id, internal_view_info);
        debug!("Image view registered: handle={:?}", handle);

        Ok(handle)
    }

    /// Creates an image, uploads data from a buffer, and transitions layout.
    pub fn create_image_init(
        &self,
        create_info: &vk::ImageCreateInfo, // Must have usage TRANSFER_DST
        location: MemoryLocation,          // Usually GpuOnly for textures
        aspect_flags: vk::ImageAspectFlags,
        data: &[u8],
    ) -> Result<ImageHandle> {
        if data.is_empty() {
            return Err(ResourceManagerError::Other(
                "Cannot create image with empty data".to_string(),
            ));
        }
        if !create_info
            .usage
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err(ResourceManagerError::Other(
                "Image create info must include TRANSFER_DST usage for init".to_string(),
            ));
        }
        // It's okay if initialLayout is not UNDEFINED, we override it for the creation step
        // but the user might have specified it for other reasons. We just ensure the
        // internal creation uses UNDEFINED.
        // if create_info.initial_layout != vk::ImageLayout::UNDEFINED {
        //    warn!(
        //       "create_image_init expects initial_layout UNDEFINED, overriding."
        //   );
        // }

        let data_size = data.len() as vk::DeviceSize;
        debug!(
            "Creating image with init data: size={}, format={:?}, extent={:?}",
            data_size, create_info.format, create_info.extent
        );

        // --- Corrected Flow ---

        // 1. Create Staging Buffer (CPU accessible for copy)
        //    Use create_buffer directly, then map and copy.
        let staging_handle = self.create_buffer(
            data_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu, // Mapped memory for upload
        )?;
        // Map & Copy data to staging buffer
        {
            // Scope for buffer info and mapping pointer
            let staging_info = self.get_buffer_info(staging_handle)?; // Lock buffers map
            let mapping = staging_info
                .mapped_ptr
                .ok_or(ResourceManagerError::MappingFailed)?;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), mapping, data.len());
            }
            // Optional: Flush if memory is not HOST_COHERENT
            // Check allocation properties if needed and flush
            trace!("Data copied to staging buffer handle {:?}", staging_handle);
        } // staging_info goes out of scope, unlocks buffers map

        // 2. Create Destination Image (ensure layout is UNDEFINED initially)
        let mut final_create_info = *create_info;
        final_create_info.initial_layout = vk::ImageLayout::UNDEFINED; // MUST start as undefined for transition
        let image_handle = self.create_image(&final_create_info, location, aspect_flags)?;

        // 3. Perform Layout Transition (Undefined -> TransferDst) & Copy
        {
            let transfer_setup_locked = self.transfer_setup.lock()?;
            let buffers_locked = self.buffers.lock()?;
            let mut images_locked = self.images.lock()?; // Mut needed to update layout

            let image_internal = images_locked
                .get_mut(&image_handle.0) // Get mut ref
                .ok_or(ResourceManagerError::HandleNotFound(image_handle.0))?;
            // Get the *correct* staging buffer info
            let staging_internal = buffers_locked
                .get(&staging_handle.0) // Use the handle created in step 1
                .ok_or(ResourceManagerError::HandleNotFound(staging_handle.0))?;

            let subresource_layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(aspect_flags)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(create_info.array_layers);

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0) // 0 means tightly packed
                .buffer_image_height(0) // 0 means tightly packed
                .image_subresource(subresource_layers)
                .image_offset(vk::Offset3D::default())
                .image_extent(create_info.extent);

            trace!("Submitting image transition and copy command...");
            unsafe {
                Self::submit_commands_and_wait(self, &transfer_setup_locked, |cmd| {
                    // Barrier 1: Undefined -> TransferDstOptimal
                    let (src_access, dst_access, src_stage, dst_stage) = Self::get_barrier_params(
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    );
                    let barrier1 = vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image_internal.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(aspect_flags)
                                .base_mip_level(0)
                                .level_count(create_info.mip_levels)
                                .base_array_layer(0)
                                .layer_count(create_info.array_layers),
                        )
                        .src_access_mask(src_access)
                        .dst_access_mask(dst_access);

                    self.device.raw().cmd_pipeline_barrier(
                        cmd,
                        src_stage,
                        dst_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier1],
                    );

                    // Copy Command (using correct staging buffer)
                    self.device.raw().cmd_copy_buffer_to_image(
                        cmd,
                        staging_internal.buffer, // Use buffer from staging_handle
                        image_internal.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL, // Layout during copy
                        &[copy_region],
                    );

                    // Barrier 2: TransferDstOptimal -> ShaderReadOnlyOptimal
                    let final_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    let (src_access, dst_access, src_stage, dst_stage) = Self::get_barrier_params(
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        final_layout,
                    );
                    let barrier2 = vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(final_layout)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image_internal.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(aspect_flags)
                                .base_mip_level(0)
                                .level_count(create_info.mip_levels)
                                .base_array_layer(0)
                                .layer_count(create_info.array_layers),
                        )
                        .src_access_mask(src_access)
                        .dst_access_mask(dst_access);

                    self.device.raw().cmd_pipeline_barrier(
                        cmd,
                        src_stage,
                        dst_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier2],
                    );

                    Ok(()) // Return Ok from the closure
                })?; // End submit_commands_and_wait
            } // Locks released (transfer_setup, buffers, images)

            drop(images_locked);

            let mut images_locked_update = self.images.lock()?; // Use different name to avoid confusion
            if let Some(info) = images_locked_update.get_mut(&image_handle.0) {
                info.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL; // Update to final layout
                trace!(
                    "Image {:?} layout updated internally to {:?}",
                    image_handle,
                    info.layout
                );
            } else {
                warn!(
                    "Image {:?} disappeared after creation during init!",
                    image_handle
                );
            }
        } // Scope for locks ends here

        // 4. Cleanup staging buffer (the one created in step 1)
        self.destroy_buffer(staging_handle)?; // Destroy the correct handle
        debug!(
            "Staging buffer destroyed for image init: handle={:?}",
            staging_handle
        ); // Log handle

        tracing::info!(
            "Image created and initialized successfully: handle={:?}",
            image_handle
        );
        Ok(image_handle)
    }

    /// Gets non-owning information about a sampler.
    pub fn get_sampler_info(&self, handle: SamplerHandle) -> Result<SamplerInfo> {
        let samplers_map = self
            .samplers
            .lock()
            .map_err(|_| ResourceManagerError::Other("Sampler map mutex poisoned".to_string()))?;
        samplers_map
            .get(&handle.0)
            .map(|internal| SamplerInfo {
                handle: internal.handle,
                sampler: internal.sampler,
                desc: internal.desc.clone(),
            })
            .ok_or(ResourceManagerError::HandleNotFound(handle.0))
    }

    /// Gets or creates a Vulkan sampler based on the description.
    /// Uses caching to avoid creating duplicate samplers.
    pub fn get_or_create_sampler(&self, desc: &SamplerDesc) -> Result<SamplerHandle> {
        let mut cache = self.sampler_cache_by_desc.lock()?;
        if let Some(handle) = cache.get(desc) {
            if self.samplers.lock()?.contains_key(&handle.0) {
                trace!("Using cached sampler for desc: {:?}", desc);
                return Ok(*handle);
            } else {
                warn!(
                    "Sampler handle {:?} found in cache but not main map. Removing from cache.",
                    handle
                );
                cache.remove(desc);
            }
        }

        drop(cache);

        trace!("Creating a new sampler for desc: {:?}", desc);
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(desc.mag_filter)
            .min_filter(desc.min_filter)
            .mipmap_mode(desc.mipmap_mode)
            .address_mode_u(desc.address_mode_u)
            .address_mode_v(desc.address_mode_v)
            .address_mode_w(desc.address_mode_w)
            .mip_lod_bias(0.0)
            .anisotropy_enable(false) // TODO: Expose anisotropy in SamplerDesc?
            .max_anisotropy(1.0)
            .compare_enable(false) // TODO: Expose compare op?
            .compare_op(vk::CompareOp::ALWAYS)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE) // TODO: Allow setting max LOD (e.g., for mipmapping)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK) // TODO: Expose border color?
            .unnormalized_coordinates(false);

        let vk_sampler = unsafe { self.device.raw().create_sampler(&sampler_info, None)? };

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let handle = SamplerHandle(id);

        let internal_info = InternalSamplerInfo {
            device: self.device.clone(),
            sampler: vk_sampler,
            handle,
            desc: desc.clone(),
        };

        self.samplers.lock()?.insert(id, internal_info);
        self.sampler_cache_by_desc
            .lock()?
            .insert(desc.clone(), handle);

        debug!("Sampler created successfully: handle={:?}", handle);
        Ok(handle)
    }

    pub fn destroy_sampler(&self, handle: SamplerHandle) -> Result<()> {
        debug!("Requesting destroy for sampler handle {:?}", handle);
        let mut samplers_map = self.samplers.lock()?;
        if let Some(internal_info) = samplers_map.remove(&handle.0) {
            self.sampler_cache_by_desc
                .lock()?
                .remove(&internal_info.desc);
            debug!("Sampler handle {:?} removed for destruction.", handle);
            Ok(())
        } else {
            warn!(
                "Attempted to destroy non-existent sampler handle {:?}",
                handle
            );
            Err(ResourceManagerError::HandleNotFound(handle.0))
        }
    }

    /// Loads a texture from a file path or embedded data using glTF info.
    /// Handles image decoding, resource creation, upload and caching.
    pub fn load_texture(
        &self,
        gltf_image: &gltf::Image,
        gltf_source: &gltf::image::Source,
        base_path: &Path,
        buffers: &[gltf::buffer::Data],
        usage: vk::ImageUsageFlags,
    ) -> Result<Arc<Texture>> {
        let cache_key_path: Option<PathBuf> = match gltf_source {
            gltf::image::Source::View { view, mime_type } => todo!(),
            gltf::image::Source::Uri { uri, mime_type } => {
                let image_path = base_path.join(uri);
                match fs::canonicalize(&image_path) {
                    Ok(canon_path) => Some(canon_path),
                    Err(e) => {
                        warn!(
                            "Failed to canonicalize image path {:?}: {} Skipping cache lookup",
                            image_path, e
                        );
                        None
                    }
                }
            }
        };

        if let Some(ref path_key) = cache_key_path {
            let uri_cache = self.texture_cache_uri.lock()?;
            if let Some(cached_texture) = uri_cache.get(path_key) {
                if self.images.lock()?.contains_key(&cached_texture.handle.0) {
                    trace!("Using cached texture (URI): {:?}", path_key);
                    return Ok(cached_texture.clone());
                } else {
                    warn!(
                        "Texture hadnle {:?} found in URI cache but not in main map. Will reload.",
                        cached_texture.handle
                    );
                }
            }
        }

        let (image_data, _format_hint) = match gltf_source {
            gltf::image::Source::Uri { uri, mime_type } => {
                let image_path = base_path.join(uri);
                debug!("Loading texture from URI: {:?}", image_path);
                let bytes = fs::read(&image_path).map_err(|e| {
                    error!("Failed to read image file {:?}: {}", image_path, e);
                    ResourceManagerError::Io(e)
                })?;
                (bytes, mime_type.map(|s| s.to_string()))
            }
            gltf::image::Source::View { view, mime_type } => {
                debug!(
                    "Loading texture from buffer view: index={}, offset={}, length={}",
                    view.buffer().index(),
                    view.offset(),
                    view.length()
                );
                let buffer_data = &buffers[view.buffer().index()];
                let start = view.offset();
                let end = start + view.length();
                if end > buffer_data.len() {
                    return Err(ResourceManagerError::Other(format!(
                        "Buffer view out of bounds for image: view index {}, buffer index {}",
                        view.index(),
                        view.buffer().index()
                    )));
                }
                let bytes: Vec<u8> = buffer_data[start..end].to_vec(); // Clone data
                (bytes, Some(mime_type.to_string())) // mime_type is required for View
            }
        };

        let img = image::load_from_memory(&image_data)?;

        let rgba_image = img.to_rgba8();
        let (width, height) = rgba_image.dimensions();
        let raw_pixels = rgba_image.into_raw();

        let vk_format = vk::Format::R8G8B8A8_SRGB;

        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(extent)
            .mip_levels(1) // TODO: Add mipmap generation
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL) // Textures should be optimal
            .usage(usage | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED) // Ensure needed usages
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED); // create_image_init handles transition

        let image_handle = self.create_image_init(
            &image_create_info,
            MemoryLocation::GpuOnly, // Textures usually live on GPU
            vk::ImageAspectFlags::COLOR,
            &raw_pixels,
        )?;

        let app_texture = Arc::new(Texture {
            handle: image_handle,
            format: vk_format,
            extent,
        });

        if let Some(path_key) = cache_key_path {
            self.texture_cache_uri
                .lock()?
                .insert(path_key, app_texture.clone());
            trace!("Texture added to URI cache.");
        }

        Ok(app_texture)
    }

    /// Transitions the layout of an image using a command buffer.
    /// Updates the internal layout state.
    pub fn transition_image_layout(
        &self,
        image_handle: ImageHandle,
        new_layout: vk::ImageLayout,
    ) -> Result<()> {
        let transfer_setup_locked = self.transfer_setup.lock()?;
        let mut images_locked = self.images.lock()?;

        let internal_info = images_locked
            .get_mut(&image_handle.0)
            .ok_or(ResourceManagerError::HandleNotFound(image_handle.0))?;

        let old_layout = internal_info.layout;
        if old_layout == new_layout {
            trace!(
                "Image {:?} already in layout {:?}. Skipping transition.",
                image_handle,
                new_layout
            );
            return Ok(());
        }

        trace!(
            "Transitioning image {:?} layout {:?} -> {:?}",
            image_handle,
            old_layout,
            new_layout
        );

        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            || old_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        unsafe {
            Self::submit_commands_and_wait(self, &transfer_setup_locked, |cmd| {
                let (src_access_mask, dst_access_mask, src_stage, dst_stage) =
                    Self::get_barrier_params(old_layout, new_layout);

                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(old_layout)
                    .new_layout(new_layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(internal_info.image)
                    .subresource_range(subresource_range)
                    .src_access_mask(src_access_mask)
                    .dst_access_mask(dst_access_mask);

                self.device.raw().cmd_pipeline_barrier(
                    cmd,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
                Ok(())
            })?;
        }

        internal_info.layout = new_layout;
        debug!(
            "Image {:?} layout transitioned to {:?}  🏳️‍⚧️",
            image_handle, new_layout
        );
        Ok(())
    }

    fn get_barrier_params(
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) -> (
        vk::AccessFlags,
        vk::AccessFlags,
        vk::PipelineStageFlags,
        vk::PipelineStageFlags,
    ) {
        let src_access_mask;
        let dst_access_mask;
        let src_stage;
        let dst_stage;

        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
                src_access_mask = vk::AccessFlags::empty();
                dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                src_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
                dst_stage = vk::PipelineStageFlags::TRANSFER;
            }
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
                src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                dst_access_mask = vk::AccessFlags::SHADER_READ;
                src_stage = vk::PipelineStageFlags::TRANSFER;
                dst_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
            }
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
                src_access_mask = vk::AccessFlags::empty();
                dst_access_mask = vk::AccessFlags::SHADER_READ;
                src_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
                dst_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
            }
            _ => {
                warn!(
                    "Unsupported layout trasnition: {:?} -> {:?}",
                    old_layout, new_layout
                );
                src_access_mask = vk::AccessFlags::MEMORY_WRITE;
                dst_access_mask = vk::AccessFlags::MEMORY_READ;
                src_stage = vk::PipelineStageFlags::ALL_COMMANDS;
                dst_stage = vk::PipelineStageFlags::ALL_COMMANDS;
            }
        }

        (src_access_mask, dst_access_mask, src_stage, dst_stage)
    }

    /// Destroys a buffer and frees its memory.
    pub fn destroy_buffer(&self, handle: BufferHandle) -> Result<()> {
        debug!("Requesting destroy for buffer handle {:?}", handle);
        let mut buffers_map = self.buffers.lock()?;
        // Remove the entry. The Drop impl of InternalBufferInfo handles the cleanup.
        if buffers_map.remove(&handle.0).is_some() {
            debug!("Buffer handle {:?} removed for destruction.", handle);
            Ok(())
        } else {
            warn!(
                "Attempted to destroy non-existent buffer handle {:?}",
                handle
            );
            Err(ResourceManagerError::HandleNotFound(handle.0))
        }
    }

    /// Destroys an image, its view, and frees its memory.
    pub fn destroy_image(&self, handle: ImageHandle) -> Result<()> {
        debug!("Requesting destroy for image handle {:?}", handle);
        let mut images_map = self.images.lock()?;
        // Remove the entry. The Drop impl of InternalImageInfo handles the cleanup.
        if images_map.remove(&handle.0).is_some() {
            debug!("Image handle {:?} removed for destruction.", handle);
            Ok(())
        } else {
            warn!(
                "Attempted to destroy non-existent image handle {:?}",
                handle
            );
            Err(ResourceManagerError::HandleNotFound(handle.0))
        }
    }

    /// Gets non-owning information about a buffer.
    pub fn get_buffer_info(&self, handle: BufferHandle) -> Result<BufferInfo> {
        let buffers_map = self.buffers.lock()?;
        buffers_map
            .get(&handle.0)
            .map(|internal| {
                let mapped_ptr = internal
                    .allocation
                    .as_ref()
                    .and_then(|a| a.mapped_ptr().map(|p| p.as_ptr() as *mut u8));

                BufferInfo {
                    handle: internal.handle,
                    buffer: internal.buffer,
                    size: internal.size,
                    usage: internal.usage,
                    mapped_ptr,
                }
            })
            .ok_or(ResourceManagerError::HandleNotFound(handle.0))
    }

    pub fn get_image_info(&self, handle: ImageHandle) -> Result<ImageInfo> {
        let images_map = self
            .images
            .lock()
            .map_err(|_| ResourceManagerError::Other("Image map mutex poisoned".to_string()))?;
        images_map
            .get(&handle.0)
            .map(|internal| {
                let mapped_ptr = internal
                    .allocation
                    .as_ref()
                    .and_then(|a| a.mapped_ptr().map(|p| p.as_ptr() as *mut u8));

                ImageInfo {
                    handle: internal.handle,
                    image: internal.image,
                    default_view_handle: internal.default_view_handle, // Return handle
                    format: internal.format,
                    extent: internal.extent,
                    usage: internal.usage,
                    layout: internal.layout,
                    mapped_ptr,
                }
            })
            .ok_or(ResourceManagerError::HandleNotFound(handle.0))
    }

    /// Gets non-owning information about a specific image view.
    pub fn get_image_view_info(&self, handle: ImageViewHandle) -> Result<ImageViewInfo> {
        let views_map = self.image_views.lock().map_err(|_| {
            ResourceManagerError::Other("Image view map mutex poisoned".to_string())
        })?;
        views_map
            .get(&handle.0)
            .map(|internal| ImageViewInfo {
                handle: internal.handle,
                view: internal.view,
            })
            .ok_or(ResourceManagerError::HandleNotFound(handle.0)) // Use handle.0
    }

    /// Explicitly waits for the device to be idle. Useful before shutdown.
    pub fn wait_device_idle(&self) -> Result<(), ResourceManagerError> {
        self.device
            .wait_idle()
            .map_err(|e| ResourceManagerError::Other(format!("Device wait idle failed: {}", e)))
    }
}

impl Drop for ResourceManager {
    fn drop(&mut self) {
        debug!("Destroying ResourceManager...");
        // Wait for idle BEFORE locking/clearing maps
        if let Err(e) = unsafe { self.device.raw().device_wait_idle() } {
            error!(
                "Failed to wait for device idle during ResourceManager drop: {}",
                e
            );
            // Proceeding, but cleanup might be unsafe
        }

        // Clear resource maps. This triggers the Drop impl for each Internal*Info.
        if let Ok(mut buffers_map) = self.buffers.lock() {
            debug!("Clearing {} buffer entries...", buffers_map.len());
            buffers_map.clear();
        } else {
            error!("Buffer map mutex poisoned during drop.");
        }

        if let Ok(mut images_map) = self.images.lock() {
            debug!("Clearing {} image entries...", images_map.len());
            images_map.clear();
        } else {
            error!("Image map mutex poisoned during drop.");
        }

        if let Ok(mut samplers_map) = self.samplers.lock() {
            debug!("Clearing {} sampler entries...", samplers_map.len());
            samplers_map.clear();
        } else {
            error!("Sampler map mutex poisoned during drop.");
        }

        // Clear caches (Arc drops will happen naturally)
        if let Ok(mut sampler_cache) = self.sampler_cache_by_desc.lock() {
            sampler_cache.clear();
        } else {
            error!("Sampler cache mutex poisoned during drop.");
        }
        if let Ok(mut texture_cache) = self.texture_cache_uri.lock() {
            texture_cache.clear();
        } else {
            error!("Texture URI cache mutex poisoned during drop.");
        }
        // Clear other caches...

        // Destroy transfer setup resources
        if let Ok(setup) = self.transfer_setup.lock() {
            debug!("Destroying TransferSetup resources...");
            unsafe {
                self.device
                    .raw()
                    .destroy_command_pool(setup.command_pool, None);
            }
            debug!("TransferSetup resources destroyed.");
        } else {
            error!("TransferSetup mutex poisoned during drop.");
        }

        // Allocator Drop: gpu-allocator's Allocator doesn't do anything in drop.
        // Memory is freed via allocator.free() called by Internal*Info drops.
        // The Arc<Mutex<Allocator>> will be dropped when the last reference goes away.

        debug!("ResourceManager destroyed.");
    }
}
