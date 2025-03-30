use std::sync::{Arc, Mutex};

use ash::{vk, Device as AshDevice};

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

    /// Submits command buffers to the queue using the provided device handle.
    ///
    /// This method acquires an internal lock for the duration of the submission call
    /// to prevent concurrent `vkQueueSubmit` calls on the same queue from this wrapper.
    ///
    /// # Arguments
    /// * `submit_device_raw` - The `ash::Device` handle corresponding to the device that owns the resources in `submits` and the `signal_fence`.
    /// * `submits` - A slice of `vk::SubmitInfo` describing the work to submit.
    /// * `signal_fence` - An optional `Fence` to signal when the submission completes. The fence must have been created with the same logical device as `submit_device_raw`.
    ///
    /// # Safety
    /// - `submit_device_raw` must be the correct, valid `ash::Device` handle associated with the resources being submitted.
    /// - The command buffers and synchronization primitives within `submits` must be valid and owned by the same logical device as `submit_device_raw`.
    /// - The `signal_fence`, if provided, must be valid, unsignaled, and owned by the same logical device as `submit_device_raw`.
    pub unsafe fn submit(
        &self,
        submit_device_raw: &AshDevice, // <<< Accept the ash::Device to use
        submits: &[vk::SubmitInfo],
        signal_fence: Option<&Fence>,
    ) -> Result<()> {
        debug_assert!(
            self.device.raw().handle() == submit_device_raw.handle(),
            "Queue::submit called with an ash::Device from a different logical VkDevice than the queue belongs to!"
        );
        // Optional: Check fence device consistency
        if let Some(fence) = signal_fence {
            debug_assert!(
                 fence.device().raw().handle() == submit_device_raw.handle(),
                 "Fence passed to Queue::submit belongs to a different logical device than submit_device_raw!"
             );
        }

        let fence_handle = signal_fence.map_or(vk::Fence::null(), |f| f.handle());

        // Keep the lock for thread-safety on the VkQueue object itself
        let _lock = self.submit_lock.lock();

        tracing::trace!(
            "Submitting {} batch(es) to queue family {}",
            submits.len(),
            self.family_index
        );

        // Use the EXPLICITLY PASSED submit_device_raw for the Vulkan call
        submit_device_raw.queue_submit(self.queue, submits, fence_handle)?;

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
