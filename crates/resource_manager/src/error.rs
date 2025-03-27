use ash::vk;
use thiserror::Error;

/// Error type for the resource_manager crate.
#[derive(Error, Debug)]
pub enum ResourceManagerError {
    #[error("Vulkan API error: {0}")]
    VulkanError(#[from] vk::Result),

    #[error("GPU allocation error: {0}")]
    AllocationError(#[from] gpu_allocator::AllocationError),

    #[error("Resource handle {0} not found")]
    HandleNotFound(u64),

    #[error("Failed to map buffer memory")]
    MappingFailed,

    #[error("Buffer is not CPU visible or mapped")]
    NotMapped,

    #[error("Failed to find suitable memory type")]
    NoSuitableMemoryType,

    #[error("Failed to find a queue supporting transfer operations")]
    NoTransferQueue,

    #[error("Staging transfer failed: {0}")]
    TransferFailed(String),

    #[error("Resource lock poisoned: {0}")]
    LockPoisoned(String),

    #[error("Error occurred in GfxHal: {0}")]
    GfxHalError(#[from] gfx_hal::error::GfxHalError),

    #[error("An unexpected error occurred: {0}")]
    Other(String),
}

// Implement conversion from Lock Poison errors
impl<T> From<std::sync::PoisonError<T>> for ResourceManagerError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        ResourceManagerError::LockPoisoned(e.to_string())
    }
}

pub type Result<T, E = ResourceManagerError> = std::result::Result<T, E>;
