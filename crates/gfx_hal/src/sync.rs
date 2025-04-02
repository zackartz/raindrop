use std::{sync::Arc, time::Duration};

use ash::vk;

use crate::{
    device::Device,
    error::{GfxHalError, Result},
};

/// Wraps a `vk::Fence`, used for CPU-GPU synchronization.
///
/// Owns the `vk::Fence` handle.
#[derive(Clone)]
pub struct Fence {
    device: Arc<Device>,
    fence: vk::Fence,
}

impl Fence {
    /// Creates a new `Fence`.
    ///
    /// # Arguments
    /// * `device` - The logical device.
    /// * `signaled` - If true, the fence is created in the signaled state.
    pub fn new(device: Arc<Device>, signaled: bool) -> Result<Self> {
        let create_flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let create_info = vk::FenceCreateInfo::default().flags(create_flags);
        let fence = unsafe { device.raw().create_fence(&create_info, None)? };
        tracing::trace!("Created Fence (signaled: {})", signaled);
        Ok(Self { device, fence })
    }

    /// Returns the device used by the fence.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Waits for the fence to become signaled.
    ///
    /// # Arguments
    /// * `timeout` - Maximum duration to wait. `None` waits indefinitely.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<()> {
        let timeout_ns = timeout.map_or(u64::MAX, |d| d.as_nanos() as u64);
        tracing::trace!("Waiting for Fence with timeout: {:?}", timeout);
        let fences = [self.fence];
        match unsafe { self.device.raw().wait_for_fences(&fences, true, timeout_ns) } {
            Ok(_) => {
                tracing::trace!("Fence signaled.");
                Ok(())
            }
            Err(vk::Result::TIMEOUT) => {
                tracing::trace!("Fence wait timed out.");
                Err(GfxHalError::VulkanError(vk::Result::TIMEOUT)) // Return timeout error
            }
            Err(e) => Err(GfxHalError::VulkanError(e)),
        }
    }

    /// Resets the fence to the unsignaled state.
    /// Must only be called when the fence is not in use by pending GPU work.
    pub fn reset(&self) -> Result<()> {
        tracing::trace!("Resetting Fence.");
        let fences = [self.fence];
        unsafe { self.device.raw().reset_fences(&fences)? };
        Ok(())
    }

    /// Checks the current status of the fence without waiting.
    /// Returns `Ok(true)` if signaled, `Ok(false)` if unsignaled.
    pub fn status(&self) -> Result<bool> {
        match unsafe { self.device.raw().get_fence_status(self.fence) } {
            Ok(signaled) => Ok(signaled),
            // NOT_READY means unsignaled, not an error in this context
            Err(vk::Result::NOT_READY) => Ok(false),
            Err(e) => Err(GfxHalError::VulkanError(e)),
        }
    }

    /// Gets the raw `vk::Fence` handle.
    pub fn handle(&self) -> vk::Fence {
        self.fence
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        tracing::trace!("Destroying fence...");
        unsafe {
            self.device.raw().destroy_fence(self.fence, None);
        }
        tracing::trace!("Fence destroyed.")
    }
}

/// Wraps a `vk::Semaphore`, used for GPU-GPU synchronization (within or across queues).
///
/// Owns the `vk::Semaphore` handle.
pub struct Semaphore {
    device: Arc<Device>,
    semaphore: vk::Semaphore,
}

impl Semaphore {
    /// Creates a new `Semaphore`.
    pub fn new(device: Arc<Device>) -> Result<Self> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let semaphore = unsafe { device.raw().create_semaphore(&create_info, None)? };
        tracing::trace!("Created Semaphore.");
        Ok(Self { device, semaphore })
    }

    /// Gets the raw `vk::Semaphore` handle.
    pub fn handle(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        tracing::trace!("Destroying Semaphore...");
        unsafe {
            self.device.raw().destroy_semaphore(self.semaphore, None);
        }
        tracing::trace!("Semaphore destroyed.");
    }
}

/// Wraps a `vk::Event`, used for fine-grained GPU-GPU or GPU-Host synchronization.
///
/// Owns the `vk::Event` handle.
pub struct Event {
    device: Arc<Device>,
    event: vk::Event,
}

impl Event {
    /// Creates a new `Event`.
    pub fn new(device: Arc<Device>) -> Result<Self> {
        let create_info = vk::EventCreateInfo::default();
        let event = unsafe { device.raw().create_event(&create_info, None)? };
        tracing::trace!("Created Event.");
        Ok(Self { device, event })
    }

    /// Sets the event from the host (CPU).
    pub fn set(&self) -> Result<()> {
        tracing::trace!("Setting Event from host.");
        unsafe { self.device.raw().set_event(self.event)? };
        Ok(())
    }

    /// Resets the event from the host (CPU).
    pub fn reset(&self) -> Result<()> {
        tracing::trace!("Resetting Event from host.");
        unsafe { self.device.raw().reset_event(self.event)? };
        Ok(())
    }

    /// Checks the status of the event from the host (CPU).
    /// Returns `Ok(true)` if set, `Ok(false)` if reset.
    pub fn status(&self) -> Result<bool> {
        let res = unsafe { self.device.raw().get_event_status(self.event) }?;
        Ok(res)
    }

    /// Gets the raw `vk::Event` handle.
    pub fn handle(&self) -> vk::Event {
        self.event
    }

    // Note: Setting/resetting/waiting on events from the GPU involves
    // vkCmdSetEvent, vkCmdResetEvent, vkCmdWaitEvents within command buffers.
    // These are not wrapped here but would be used via device.raw() when
    // recording command buffers.
}

impl Drop for Event {
    fn drop(&mut self) {
        tracing::trace!("Destroying Event...");
        unsafe {
            self.device.raw().destroy_event(self.event, None);
        }
        tracing::trace!("Event destroyed.");
    }
}
