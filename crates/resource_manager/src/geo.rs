use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use tracing::{debug, trace};

use crate::{BufferHandle, ResourceManager, ResourceManagerError, Result};

// Helper to safely get a byte slice from structured data
unsafe fn as_byte_slice<T: Sized>(data: &[T]) -> &[u8] {
    std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
}

/// Represents geometry data (verticies and indicies) stored in GPU buffers managed by
/// ResourceManager. Handles automatic cleanup via a `Drop` implementation.
#[derive(Clone)]
pub struct Geometry {
    resource_manager: Arc<ResourceManager>,
    pub vertex_buffer: BufferHandle,
    pub index_buffer: BufferHandle,
    pub index_count: u32,
}

impl Geometry {
    /// Creates new GPU buffers for the given vetex and index data using `ResourceManager`.
    ///
    /// # Arguments
    ///
    /// * `resource_manager` - An Arc reference to the ResourceManager.
    /// * `vertices` - A slice of vertex data.
    /// * `indices` - A slice of index data (u32)
    ///
    /// # Errors
    ///
    /// Returns a new `ResourceManagerError` if buffer creation or data upload fails.
    pub fn new<V: Sized + Copy>(
        resource_manager: Arc<ResourceManager>,
        vertices: &[V],
        indicies: &[u32],
    ) -> Result<Self> {
        trace!(
            "Creating Geometry: {} vertices, {} indicies",
            vertices.len(),
            indicies.len()
        );

        if vertices.is_empty() || indicies.is_empty() {
            return Err(ResourceManagerError::Other(
                "Cannot create Geometry with empty vertices or indicies.".to_string(),
            ));
        }

        let vertex_buffer = resource_manager.create_buffer_init(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::GpuOnly,
            unsafe { as_byte_slice(vertices) },
        )?;
        trace!("Vertex buffer created: handle={:?}", vertex_buffer);

        let index_buffer = resource_manager.create_buffer_init(
            vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryLocation::GpuOnly,
            unsafe { as_byte_slice(indicies) },
        )?;
        trace!("Index buffer created: handle={:?}", index_buffer);

        let index_count = indicies.len() as u32;

        debug!(
            "Geometry created successfully: VB={:?}, IB={:?}, Indices={}",
            vertex_buffer, index_buffer, index_count
        );

        Ok(Self {
            resource_manager,
            vertex_buffer,
            index_buffer,
            index_count,
            // vertex_count,
        })
    }

    /// Binds the vertex and index buffers for drawing.
    ///
    /// # Arguments
    ///
    /// * `device` - Raw `ash::Device` handle.
    /// * `command_buffer` - The command buffer to record binding commands into.
    ///
    /// # Errors
    ///
    /// Returns `ResourceManagerError` if buffer info cannot be retrieved.
    pub fn bind(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) -> Result<()> {
        trace!(
            "Binding geometry: VB={:?}, IB={:?}",
            self.vertex_buffer,
            self.index_buffer
        );
        // Get buffer info (locks resource manager map briefly)
        let vb_info = self.resource_manager.get_buffer_info(self.vertex_buffer)?;
        let ib_info = self.resource_manager.get_buffer_info(self.index_buffer)?;

        let vk_vertex_buffers = [vb_info.buffer];
        let offsets = [0_u64]; // Use vk::DeviceSize (u64)

        unsafe {
            device.cmd_bind_vertex_buffers(
                command_buffer,
                0, // binding = 0
                &vk_vertex_buffers,
                &offsets,
            );
            device.cmd_bind_index_buffer(
                command_buffer,
                ib_info.buffer,
                0, // offset = 0
                vk::IndexType::UINT32,
            );
        }
        Ok(())
    }

    /// Binds the geometry buffers and issues an indexed draw command.
    ///
    /// # Arguments
    ///
    /// * `device` - Raw `ash::Device` handle.
    /// * `command_buffer` - The command buffer to record commands into.
    ///
    /// # Errors
    ///
    /// Returns `ResourceManagerError` if binding fails.
    pub fn draw(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) -> Result<()> {
        self.bind(device, command_buffer)?; // Bind first
        trace!("Drawing geometry: {} indices", self.index_count);
        unsafe {
            device.cmd_draw_indexed(
                command_buffer,
                self.index_count, // Use stored index count
                1,                // instance_count
                0,                // first_index
                0,                // vertex_offset
                0,                // first_instance
            );
        }
        Ok(())
    }
}

impl Drop for Geometry {
    fn drop(&mut self) {
        debug!(
            "Dropping Geometry: VB={:?}, IB={:?}",
            self.vertex_buffer, self.index_buffer
        );
        // Request destruction from the resource manager.
        // Ignore errors during drop, but log them.
        if let Err(e) = self.resource_manager.destroy_buffer(self.vertex_buffer) {
            tracing::error!(
                "Failed to destroy vertex buffer {:?} during Geometry drop: {}",
                self.vertex_buffer,
                e
            );
        }

        if let Err(e) = self.resource_manager.destroy_buffer(self.index_buffer) {
            tracing::error!(
                "Failed to destroy index buffer {:?} during Geometry drop: {}",
                self.index_buffer,
                e
            );
        }
        // The Arc<ResourceManager> reference count decreases automatically.
    }
}
