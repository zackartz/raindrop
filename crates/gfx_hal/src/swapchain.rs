use std::sync::Arc;

use ash::khr::swapchain::Device as SwapchainLoader;
use ash::vk;

use crate::device::{self, Device};
use crate::error::{GfxHalError, Result};
use crate::surface::{self, Surface};
use crate::sync::{Fence, Semaphore};

/// Configuration for creating or recreating a `Swapchain`.
#[derive(Clone, Debug)]
pub struct SwapchainConfig {
    /// Desired number of images in the swapchain (min/max clamped by capabilities).
    pub desired_image_count: u32,
    /// Preferred surface format (e.g., `vk::Format::B8G8R8A8_SRGB`).
    pub desired_format: vk::SurfaceFormatKHR,
    /// Preferred presentation mode (e.g., `vk::PresentModeKHR::MAILBOX`).
    pub desired_present_mode: vk::PresentModeKHR,
    /// Desired usage flags for swapchain images (e.g., `vk::ImageUsageFlags::COLOR_ATTACHMENT`).
    pub image_usage: vk::ImageUsageFlags,
    /// The dimensions of the surface.
    pub extent: vk::Extent2D,
    /// Transformation to apply (usually `vk::SurfaceTransformFlagsKHR::IDENTITY`).
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    /// Alpha compositing mode (usually `vk::CompositeAlphaFlagsKHR::OPAQUE`).
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
}

/// Represents the Vulkan swapchain, managing presentation images.
///
/// Owns the `vk::SwapchainKHR`, the `ash` Swapchain loader, the swapchain images,
/// and their corresponding image views.
pub struct Swapchain {
    device: Arc<Device>,
    swapchain_loader: SwapchainLoader,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    image_count: u32,
}

impl Swapchain {
    /// Creates a new `Swapchain` or recreates an exisiting one.
    ///
    /// # Arguments
    /// * `device` - The logical device.
    /// * `surface` - The surface to present to.
    /// * `config` - Desired swapchain configuration.
    /// * `old_swapchain` - Optional handle to a previous swapchain for smoother recreation.
    ///
    /// # Safety
    /// - `device` and `surface` must be valid and compatible.
    /// - If `old_swapchain` is provided, it must be a valid handle previously created
    ///   with the same `surface`.
    /// - The caller must ensure that the `old_swapchain` (and its associated resources like
    ///   image views) are no longer in use before calling this function and are properly
    ///   destroyed *after* the new swapchain is successfully created.
    pub unsafe fn new(
        device: Arc<Device>,
        surface: Arc<Surface>,
        config: SwapchainConfig,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<Self> {
        let physical_device = device.physical_device_handle();

        let capabilities = surface.get_physical_device_surface_capabilities(physical_device)?;
        let formats = surface.get_physical_device_surface_formats(physical_device)?;
        let present_modes = surface.get_physical_device_surface_present_modes(physical_device)?;

        if formats.is_empty() || present_modes.is_empty() {
            return Err(GfxHalError::NoSuitableGpu(
                "Swapchain creation failed: No formats or present modes available.".to_string(),
            ));
        }

        let surface_format = Self::choose_surface_format(&formats, config.desired_format);
        let present_mode = Self::choose_present_mode(&present_modes, config.desired_present_mode);
        let extent = Self::choose_extent(capabilities, config.extent);
        let image_count = Self::choose_image_count(capabilities, config.desired_image_count);

        tracing::info!("Creating swapchain: Format={:?}, ColorSpace={:?}, PresentMode={:?}, Extent={:?}, ImageCount={}", surface_format.format, surface_format.color_space, present_mode, extent, image_count);

        let mut create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(config.image_usage)
            .pre_transform(config.pre_transform)
            .composite_alpha(config.composite_alpha)
            .present_mode(present_mode)
            .clipped(true);

        let queue_family_indicies = [device.graphics_queue_family_index()];
        create_info = create_info
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indicies);

        if let Some(old) = old_swapchain {
            create_info = create_info.old_swapchain(old);
            tracing::debug!("Passing old swapchain handle for recreation.");
        }

        let swapchain_loader =
            SwapchainLoader::new(surface.instance().ash_instance(), device.raw());
        let swapchain = swapchain_loader.create_swapchain(&create_info, None)?;
        tracing::info!("Swapchain created successfully.");

        let images = swapchain_loader.get_swapchain_images(swapchain)?;
        tracing::debug!("Retrieved {} swapchain images.", images.len());

        let image_views = Self::create_image_views(device.raw(), &images, surface_format.format)?;
        tracing::debug!("Created {} swapchain image views.", image_views.len());

        Ok(Self {
            device,
            swapchain_loader,
            swapchain,
            images,
            image_views,
            format: surface_format,
            extent,
            image_count,
        })
    }

    /// Acquires the next available image from the swapchain.
    ///
    /// Returns the index of the acquired image and a boolean indicating if the
    /// swapchain is suboptimal (needs recreation).
    ///
    /// # Safety
    /// - `signal_semaphore` and `signal_fence`, if provided, must be valid handles
    ///   that are not currently waited on by the GPU.
    /// - The caller must ensure proper synchronization before using the returned image index.
    pub unsafe fn acquire_next_image(
        &self,
        timeout_ns: u64,
        signal_semaphore: Option<&Semaphore>,
        signal_fence: Option<&Fence>,
    ) -> Result<(u32, bool)> {
        let semaphore_handle = signal_semaphore.map_or(vk::Semaphore::null(), |s| s.handle());
        let fence_handle = signal_fence.map_or(vk::Fence::null(), |f| f.handle());

        match self.swapchain_loader.acquire_next_image(
            self.swapchain,
            timeout_ns,
            semaphore_handle,
            fence_handle,
        ) {
            Ok((image_index, suboptimal)) => Ok((image_index, suboptimal)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(GfxHalError::SurfaceLost),
            Err(e) => Err(GfxHalError::VulkanError(e)),
        }
    }

    /// Gets the raw `vk::SwapchainKHR` handle.
    pub fn handle(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    /// Gets a reference to the `ash` Swapchain loader extension
    pub fn loader(&self) -> &SwapchainLoader {
        &self.swapchain_loader
    }

    /// Gets the chosen surface format of the swapchain.
    pub fn format(&self) -> vk::SurfaceFormatKHR {
        self.format
    }

    /// Gets the extent (dimensions) of the swapchain image
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Gets the actual number of images in the swapchain
    pub fn image_count(&self) -> u32 {
        self.image_count
    }

    /// Gets a slice containing the raw `vk::Image` handles.
    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    /// Gets a slice containing the raw `vk::ImageView` handles.
    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    fn choose_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
        desired_format: vk::SurfaceFormatKHR,
    ) -> vk::SurfaceFormatKHR {
        for format in available_formats {
            if format.format == desired_format.format
                && format.color_space == desired_format.color_space
            {
                return *format;
            }
        }

        tracing::warn!(
            "Desired swapchain format {:?}/{:?} not available. Falling back to {:?}/{:?}.",
            desired_format.format,
            desired_format.color_space,
            available_formats[0].format,
            available_formats[0].color_space
        );
        available_formats[0]
    }

    fn choose_present_mode(
        available_modes: &[vk::PresentModeKHR],
        desired_mode: vk::PresentModeKHR,
    ) -> vk::PresentModeKHR {
        if desired_mode == vk::PresentModeKHR::MAILBOX
            && available_modes.contains(&vk::PresentModeKHR::MAILBOX)
        {
            return vk::PresentModeKHR::MAILBOX;
        }

        if desired_mode == vk::PresentModeKHR::IMMEDIATE
            && available_modes.contains(&vk::PresentModeKHR::IMMEDIATE)
        {
            return vk::PresentModeKHR::IMMEDIATE;
        }

        vk::PresentModeKHR::FIFO
    }

    fn choose_extent(
        capabilities: vk::SurfaceCapabilitiesKHR,
        desired_extent: vk::Extent2D,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: desired_extent.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: desired_extent.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn choose_image_count(capabilities: vk::SurfaceCapabilitiesKHR, desired_count: u32) -> u32 {
        let mut count = desired_count.max(capabilities.min_image_count);
        if capabilities.max_image_count > 0 {
            count = count.min(capabilities.max_image_count);
        }
        count
    }

    unsafe fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Result<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                device.create_image_view(&create_info, None)
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(GfxHalError::VulkanError)
    }

    /// Destroys the image views associated with this swapchain.
    /// Called internally by Drop and potentially during recreation.
    unsafe fn destory_image_views(&mut self) {
        tracing::debug!(
            "Destroying {} swapchain image views...",
            self.image_views.len()
        );
        for view in self.image_views.drain(..) {
            self.device.raw().destroy_image_view(view, None);
        }
        tracing::debug!("Swapchain image views destroyed.")
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        tracing::debug!("Destroying swapchain...");
        unsafe {
            self.destory_image_views();
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
        tracing::debug!("Swapchain destroyed.")
    }
}
