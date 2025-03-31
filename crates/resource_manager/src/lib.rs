mod error;

use std::{
    collections::HashMap,
    hash::Hash,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

use ash::vk;
use gfx_hal::{device::Device, instance::Instance, queue::Queue};
use tracing::{debug, error, trace, warn};

pub use error::{ResourceManagerError, Result};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
    MemoryLocation,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageHandle(u64);

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
    /// Non-owning handle.
    pub image: vk::Image,
    /// Non-owning handle.
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub usage: vk::ImageUsageFlags,
    pub layout: vk::ImageLayout,
}

struct InternalBufferInfo {
    device: Arc<Device>,              // Keep device alive for Drop
    allocator: Arc<Mutex<Allocator>>, // Needed for Drop
    buffer: vk::Buffer,
    allocation: Option<Allocation>, // Option because it's taken in Drop
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    mapped_ptr: Option<*mut u8>,
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
    device: Arc<Device>,              // Keep device alive for Drop
    allocator: Arc<Mutex<Allocator>>, // Needed for Drop
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>, // Option because it's taken in Drop
    format: vk::Format,
    extent: vk::Extent3D,
    usage: vk::ImageUsageFlags,
    layout: vk::ImageLayout,
    handle: ImageHandle,
}

impl Drop for InternalImageInfo {
    fn drop(&mut self) {
        trace!("Dropping InternalImageInfo for handle {:?}", self.handle);
        // Destroy view first
        unsafe {
            self.device.raw().destroy_image_view(self.view, None);
        }
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

struct TransferSetup {
    command_pool: vk::CommandPool,
    queue: Arc<Queue>,
    fence: vk::Fence,
}

pub struct ResourceManager {
    _instance: Arc<Instance>,
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    buffers: Mutex<HashMap<u64, InternalBufferInfo>>,
    images: Mutex<HashMap<u64, InternalImageInfo>>,
    next_id: AtomicU64,
    transfer_setup: Mutex<Option<TransferSetup>>,
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

        Ok(Self {
            _instance: instance,
            device,
            allocator: Arc::new(Mutex::new(allocator)),
            buffers: Mutex::new(HashMap::new()),
            images: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            transfer_setup: Mutex::new(None),
        })
    }

    /// Gets a shared reference to the Allocator
    pub fn allocator(&self) -> Arc<Mutex<Allocator>> {
        self.allocator.clone()
    }

    /// Gets or initializes the TransferSetup resources.
    fn get_transfer_setup(&self) -> Result<TransferSetup> {
        let mut setup_guard = self.transfer_setup.lock()?;

        if let Some(setup) = setup_guard.as_ref() {
            // Simple check: Reset fence before reusing
            unsafe { self.device.raw().reset_fences(&[setup.fence])? };
            return Ok(TransferSetup {
                // Return a copy/clone
                command_pool: setup.command_pool,
                queue: setup.queue.clone(),
                fence: setup.fence,
            });
        }

        debug!("Initializing TransferSetup...");
        // Find a queue that supports transfer (prefer dedicated, fallback to graphics)
        let queue_family_index = self
            .device
            .transfer_queue_family_index()
            .or(self.device.compute_queue_family_index()) // Try compute as fallback
            .unwrap_or(self.device.graphics_queue_family_index()); // Graphics as last resort

        let queue = self.device.get_queue(queue_family_index, 0)?;

        // Create command pool for transfer commands
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT) // Hint that buffers are short-lived
            .queue_family_index(queue_family_index);
        let command_pool = unsafe { self.device.raw().create_command_pool(&pool_info, None)? };

        // Create a fence for waiting
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.raw().create_fence(&fence_info, None)? };

        let new_setup = TransferSetup {
            command_pool,
            queue,
            fence,
        };
        *setup_guard = Some(new_setup); // Store it
        debug!("TransferSetup initialized.");

        // Return a new copy for use
        Ok(TransferSetup {
            command_pool: setup_guard.as_ref().unwrap().command_pool,
            queue: setup_guard.as_ref().unwrap().queue.clone(),
            fence: setup_guard.as_ref().unwrap().fence,
        })
    }

    /// Helper to allocate, begin, end, submit, and wait for a single command buffer.
    unsafe fn submit_commands_and_wait<F>(
        &self,
        transfer_setup: &TransferSetup,
        record_fn: F,
    ) -> Result<()>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<()>,
    {
        let device = self.device.raw();

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(transfer_setup.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

        // Begin recording
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(command_buffer, &begin_info)?;

        // Record user commands
        let record_result = record_fn(command_buffer);

        // End recording (even if user function failed, to allow cleanup)
        device.end_command_buffer(command_buffer)?;

        // Check user function result *after* ending buffer
        record_result?;

        let binding = [command_buffer];
        // Submit
        let submits = [vk::SubmitInfo::default().command_buffers(&binding)];
        // Use the transfer queue and fence
        transfer_setup
            .queue
            .submit(self.device.raw(), &submits, None)?; // Submit without fence initially

        // Wait for completion using a separate wait call
        // This avoids holding the queue's internal submit lock during the wait.
        let fences = [transfer_setup.fence];
        match device.wait_for_fences(&fences, true, u64::MAX) {
            Ok(_) => {}
            Err(vk::Result::TIMEOUT) => {
                // Should not happen with u64::MAX
                warn!("Transfer fence wait timed out unexpectedly.");
                return Err(ResourceManagerError::TransferFailed(
                    "Fence wait timeout".to_string(),
                ));
            }
            Err(e) => return Err(e.into()),
        }

        // Free command buffer
        device.free_command_buffers(transfer_setup.command_pool, &[command_buffer]);

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

        let mapped_ptr = allocation.mapped_ptr().map(|p| p.as_ptr() as *mut u8);
        if mapped_ptr.is_some() {
            trace!("Buffer memory is mapped.");
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let handle = BufferHandle(id);

        let internal_info = InternalBufferInfo {
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            buffer,
            allocation: Some(allocation),
            size,
            usage,
            mapped_ptr,
            handle,
        };

        self.buffers.lock()?.insert(id, internal_info);
        debug!("Buffer created successfully: handle={:?}", handle);
        Ok(handle)
    }

    /// Creates a buffer, allocates memory, and uploads initial data using a staging buffer.
    pub fn create_buffer_init(
        &self,
        usage: vk::BufferUsageFlags, // Usage for the *final* buffer
        location: MemoryLocation,    // Memory location for the *final* buffer
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

        // 1. Create Staging Buffer
        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let staging_location = MemoryLocation::CpuToGpu; // Mapped memory for upload
        let staging_handle = self.create_buffer(size, staging_usage, staging_location)?;

        // 2. Map & Copy data to staging buffer
        {
            // Scope for buffer info and mapping pointer
            let staging_info = self.get_buffer_info(staging_handle)?;
            let mapping = staging_info
                .mapped_ptr
                .ok_or(ResourceManagerError::MappingFailed)?;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), mapping, data.len());
            }
            // If memory is not HOST_COHERENT, need to flush here:
            // let mem_range = vk::MappedMemoryRange::builder().memory(...).offset(...).size(size);
            // unsafe { self.device.raw().flush_mapped_memory_ranges(&[mem_range])? };
            trace!("Data copied to staging buffer.");
        } // staging_info goes out of scope

        // 3. Create Destination Buffer
        let final_usage = usage | vk::BufferUsageFlags::TRANSFER_DST; // Add transfer dest usage
        let dest_handle = self.create_buffer(size, final_usage, location)?;

        // 4. Record and submit transfer command
        let transfer_setup = self.get_transfer_setup()?;
        let dest_info = self.get_buffer_info(dest_handle)?; // Get info for vk::Buffer handle
        let staging_info_for_copy = self.get_buffer_info(staging_handle)?; // Get info again

        trace!("Submitting buffer copy command...");
        unsafe {
            self.submit_commands_and_wait(&transfer_setup, |cmd| {
                let region = vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(size);
                self.device.raw().cmd_copy_buffer(
                    cmd,
                    staging_info_for_copy.buffer, // Use raw handle from info struct
                    dest_info.buffer,             // Use raw handle from info struct
                    &[region],
                );
                Ok(()) // Return Ok inside the closure
            })?;
        }
        trace!("Buffer copy command finished.");

        // 5. Cleanup staging buffer
        self.destroy_buffer(staging_handle)?; // This frees memory and destroys buffer
        debug!("Staging buffer destroyed.");

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

        // Create a default image view
        // TODO: Make view creation more flexible (allow different subresource ranges, types)
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D) // Assuming 2D, adjust based on create_info
            .format(create_info.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: create_info.mip_levels,
                base_array_layer: 0,
                layer_count: create_info.array_layers,
            });
        let view = unsafe { self.device.raw().create_image_view(&view_info, None)? };
        trace!("Default image view created.");

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let handle = ImageHandle(id);

        let internal_info = InternalImageInfo {
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            image,
            view,
            allocation: Some(allocation),
            format: create_info.format,
            extent: create_info.extent,
            usage: create_info.usage,
            layout: create_info.initial_layout, // Store initial layout
            handle,
        };

        self.images.lock()?.insert(id, internal_info);
        debug!("Image created successfully: handle={:?}", handle);
        Ok(handle)
    }

    // TODO: Implement create_image_init (similar to create_buffer_init but uses vkCmdCopyBufferToImage and layout transitions)

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
            .map(|internal| BufferInfo {
                handle: internal.handle,
                buffer: internal.buffer,
                size: internal.size,
                usage: internal.usage,
                mapped_ptr: internal.mapped_ptr,
            })
            .ok_or(ResourceManagerError::HandleNotFound(handle.0))
    }

    /// Gets non-owning information about an image.
    pub fn get_image_info(&self, handle: ImageHandle) -> Result<ImageInfo> {
        let images_map = self.images.lock()?;
        images_map
            .get(&handle.0)
            .map(|internal| ImageInfo {
                handle: internal.handle,
                image: internal.image,
                view: internal.view,
                format: internal.format,
                extent: internal.extent,
                usage: internal.usage,
                layout: internal.layout, // Note: Layout tracking is basic here
            })
            .ok_or(ResourceManagerError::HandleNotFound(handle.0))
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
        // Ensure all GPU operations are finished before freeing memory/destroying resources
        if let Err(e) = self.device.wait_idle() {
            error!(
                "Failed to wait for device idle during ResourceManager drop: {}",
                e
            );
            // Proceeding with cleanup, but resources might still be in use!
        }

        // Clear resource maps. This triggers the Drop impl for each Internal*Info,
        // which frees allocations and destroys Vulkan objects.
        let mut buffers_map = self.buffers.lock().expect("mutex to not be poisoned");
        debug!("Clearing {} buffer entries...", buffers_map.len());
        buffers_map.clear();
        let mut images_map = self.images.lock().expect("mutex to not be poisoned");
        debug!("Clearing {} image entries...", images_map.len());
        images_map.clear();

        // Destroy transfer setup resources
        let mut setup_guard = self
            .transfer_setup
            .lock()
            .expect("mutex to not be poisoned");
        if let Some(setup) = setup_guard.take() {
            // take() removes it from the Option
            debug!("Destroying TransferSetup resources...");
            unsafe {
                self.device.raw().destroy_fence(setup.fence, None);
                self.device
                    .raw()
                    .destroy_command_pool(setup.command_pool, None);
            }
            debug!("TransferSetup resources destroyed.");
        }

        // The Allocator is wrapped in an Arc<Mutex<>>, so its Drop will be handled
        // when the last Arc reference (including those held by Internal*Info) is dropped.
        // gpu-allocator's Allocator Drop implementation should be empty, as memory
        // is freed via allocator.free().

        debug!("ResourceManager destroyed.");
    }
}
