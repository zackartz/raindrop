use std::{
    sync::{Arc, Mutex, MutexGuard},
    thread::Result,
};

use ash::{vk, Device};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation,
};

#[derive(Debug)]
pub enum RtError {
    VulkanError(vk::Result),
    AllocationError(gpu_allocator::AllocationError),
}

pub struct AccelerationStructure {
    accel: vk::AccelerationStructureKHR,
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
}

impl AccelerationStructure {
    fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe { device.destroy_buffer(self.buffer, None) };
        if let Some(allocation) = self.allocation.take() {
            allocator.free(allocation).unwrap();
        }
    }
}

pub struct Buffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    size: vk::DeviceSize,
}

impl Buffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> color_eyre::Result<Self> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let buffer = unsafe { device.create_buffer(&buffer_info, None) }?;

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "rt_buffer",
            requirements,
            location,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;

        Ok(Self {
            buffer,
            allocation: Some(allocation),
            size,
        })
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe { device.destroy_buffer(self.buffer, None) };
        if let Some(allocation) = self.allocation.take() {
            allocator.free(allocation).unwrap();
        }
    }
}

pub struct RaytracingBuilderKHR {
    device: Arc<Device>,
    queue_index: u32,
    allocator: Arc<Mutex<Allocator>>,
    blas: Vec<AccelerationStructure>,
    tlas: AccelerationStructure,
    command_pool: vk::CommandPool,
}

impl RaytracingBuilderKHR {
    pub fn new() -> Self {
        Self {
            device: None,
            queue_index: 0,
            allocator: None,
            blas: Vec::new(),
            tlas: AccelerationStructure::de,
        }
    }
}
