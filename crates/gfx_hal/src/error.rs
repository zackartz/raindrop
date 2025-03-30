use ash::vk;
use thiserror::Error;

/// Top-level error type for the gfx_hal crate.
#[derive(Error, Debug)]
pub enum GfxHalError {
    /// Error originating directly from a Vulkan API call.
    #[error("Vulkan API Error: {0}")]
    VulkanError(#[from] vk::Result),

    /// Error loading the Vulkan library or resolving function pointers.
    #[error("Failed to load Vulkan library: {0}")]
    LoadError(String),

    /// No suitable physical device (GPU) could be found that meets requirements.
    #[error("No suitable physical device found: {0}")]
    NoSuitableGpu(String),

    /// A required Vulkan instance or device extension is not supported.
    #[error("Missing required Vulkan extension: {0:?}")]
    MissingExtension(String),

    /// A required Vulkan feature is not supported by the physical device.
    #[error("Missing required Vulkan feature: {0}")]
    MissingFeature(String),

    /// Failed to find a suitable queue family (e.g., graphics, present).
    #[error("Could not find required queue family: {0}")]
    MissingQueueFamily(String),

    /// Error related to window system integration surface creation.
    #[error("Failed to create Vulkan surface: {0}")]
    SurfaceCreationError(vk::Result),

    /// The Vulkan surface became invalid (e.g., window resized, closed).
    #[error("Vulkan surface is no longer valid (maybe lost or out of date)")]
    SurfaceLost,

    /// Error converting a C-style string.
    #[error("Invalid C string: {0}")]
    InvalidCString(#[from] std::ffi::NulError),

    /// Error converting C string slice to Rust string slice.
    #[error("Invalid UTF-8 sequence in C string: {0}")]
    InvalidCStringUtf8(#[from] std::ffi::FromBytesWithNulError),

    /// Generic I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Error with winit windowing library.
    #[error("Winit Error: {0}")]
    WinitHandleError(#[from] winit::raw_window_handle::HandleError),

    /// Ash loader error.
    #[error("Error loading the ash entry.")]
    AshEntryError(#[from] ash::LoadingError),

    /// Poisoned Mutex
    #[error("Error from poisoned mutex: {0}")]
    MutexPoisoned(String),

    /// Placeholder for other specific errors.
    #[error("An unexpected error occurred: {0}")]
    Other(String),
}

pub type Result<T, E = GfxHalError> = std::result::Result<T, E>;

impl<T> From<std::sync::PoisonError<T>> for GfxHalError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::MutexPoisoned(e.to_string())
    }
}
