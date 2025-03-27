use std::sync::Arc;

use ash::vk;
use parking_lot::Mutex;

use crate::device::Device;
use crate::error::Result;
use crate::sync::Fence;

/// Represents a Vulkan device queue.
///
/// Holds a reference to the `Device` and the raw `vk::Queue` handle.
/// Provides methods for submitting command buffers.
pub struct Queue {
    device: Arc<Device>,
    queue: vk::Queue,
    family_index: u32,
    // Each queue submission must be externally synchronized or locked internally.
    // Using a Mutex here provides a simple internal locking per queue.
    submit_lock: Mutex<()>,
}

impl Queue {
    /// Creates a new Queue wrapper. Called internally by `Device`.
    pub(crate) fn new(device: Arc<Device>, queue: vk::Queue, family_index: u32) -> Self {
        Self {
            device,
            queue,
            family_index,
            submit_lock: Mutex::new(()),
        }
    }

    /// Gets the raw `vk::Queue` handle.
    pub fn handle(&self) -> vk::Queue {
        self.queue
    }

    /// Gets the queue family index this queue belongs to.
    pub fn family_index(&self) -> u32 {
        self.family_index
    }

    /// Gets a reference to the logical device this queue belongs to.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Submits command buffers to the queue.
    ///
    /// This method acquires an internal lock for the duration of the submission call
    /// to prevent concurrent `vkQueueSubmit` calls on the same queue from this wrapper.
    ///
    /// # Arguments
    /// * `submits` - A slice of `vk::SubmitInfo` describing the work to submit.
    /// * `signal_fence` - An optional `Fence` to signal when the submission completes.
    ///
    /// # Safety
    /// - The command buffers and synchronization primitieves within `submits` must be valid.
    /// - The `signal_fence`, if provided, must be valid and unsignaled.
    pub unsafe fn submit(
        &self,
        submits: &[vk::SubmitInfo],
        signal_fence: Option<&Fence>,
    ) -> Result<()> {
        let fence_handle = signal_fence.map_or(vk::Fence::null(), |f| f.handle());

        let _lock = self.submit_lock.lock();

        tracing::trace!(
            "Submitting {} batch(es) to queue family {}",
            submits.len(),
            self.family_index
        );
        self.device
            .raw()
            .queue_submit(self.queue, submits, fence_handle)?;
        tracing::trace!("Submission successful.");
        Ok(())
    }

    /// Waits until this queue becomes idle.
    ///
    /// This is a heavy operation and blocks the current thread.
    pub fn wait_idle(&self) -> Result<()> {
        tracing::debug!("Waiting for queue idle (family {})...", self.family_index);
        // Lock the mutex while waiting to prevent submissions during the wait?
        // Or allow submissions and let Vulkan handle it? Let Vulkan handle it.
        unsafe { self.device.raw().queue_wait_idle(self.queue)? };
        tracing::debug!("Queue idle (family {}).", self.family_index);
        Ok(())
    }

    // Note: vkQueuePresentKHR is intentionally omitted here.
    // Presentation is tightly coupled with Swapchain. It's safer to
    // have a method like `Swapchain::present(&self, queue: &Queue, ...)`
    // which internally calls `queue.device().raw().queue_present_khr(...)`
    // using the swapchain's loader.
    // If direct access is needed, it can be done with `queue.device().raw()`.
}

// Queues don't own the vk::Queue handle (the Device does), so no Drop impl.
