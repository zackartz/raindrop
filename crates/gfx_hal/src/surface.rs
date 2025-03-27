use crate::{
    error::{GfxHalError, Result},
    instance::Instance,
};

use ash::{khr::surface::Instance as SurfaceLoader, vk};
use std::sync::Arc;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// Represents a Vulkan presentation surface, tied to a window.
///
/// Owns the `vk::SurfaceKHR` handle and the `ash` Surface loader extension.
pub struct Surface {
    instance: Arc<Instance>,
    surface_loader: SurfaceLoader,
    surface: vk::SurfaceKHR,
}

impl Surface {
    /// Creates a new Vulkan `Surface`
    ///
    /// # Safety
    /// - The `window_handle_trait_obj` must provide valid window and display handles
    ///   for the lifetime of the `Surface`.
    /// - The `Instance` must outlive the `Surface`
    pub unsafe fn new(
        instance: Arc<Instance>,
        window_handle_trait_obj: &(impl HasWindowHandle + HasDisplayHandle),
    ) -> Result<Arc<Self>> {
        let surface_loader = SurfaceLoader::new(instance.entry(), instance.ash_instance());
        let surface = ash_window::create_surface(
            instance.entry(),
            instance.ash_instance(),
            window_handle_trait_obj.display_handle()?.into(),
            window_handle_trait_obj.window_handle()?.into(),
            None,
        )
        .map_err(GfxHalError::SurfaceCreationError)?;

        tracing::info!("Vulkan surface created successfully.");

        Ok(Arc::new(Self {
            instance,
            surface_loader,
            surface,
        }))
    }

    /// Gets the raw `vk::SurfaceKHR` handle.
    pub fn handle(&self) -> vk::SurfaceKHR {
        self.surface
    }

    /// Gets a reference to the `ash` Surface loader extension.
    pub fn surface_loader(&self) -> &SurfaceLoader {
        &self.surface_loader
    }

    /// Gets a reference to the `Instance` this surface belongs to
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Queries surface capabilites for a given physical device.
    ///
    /// # Safety
    /// The `physical_device` handle must be valid and compatible with this surface.
    pub unsafe fn get_physical_device_surface_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::SurfaceCapabilitiesKHR> {
        self.surface_loader
            .get_physical_device_surface_capabilities(physical_device, self.surface)
            .map_err(GfxHalError::VulkanError)
    }

    /// Queries supported surface formats for a given physical device.
    ///
    /// # Safety
    /// The `physical_device` handle must be valid and compatible with this surface.
    pub unsafe fn get_physical_device_surface_formats(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::SurfaceFormatKHR>> {
        self.surface_loader
            .get_physical_device_surface_formats(physical_device, self.surface)
            .map_err(GfxHalError::VulkanError)
    }

    /// Queries supported present modes for a given physical device.
    ///
    /// # Safety
    /// The `physical_device` handle must be valid and compatible with this surface.
    pub unsafe fn get_physical_device_surface_present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::PresentModeKHR>> {
        self.surface_loader
            .get_physical_device_surface_present_modes(physical_device, self.surface)
            .map_err(GfxHalError::VulkanError)
    }

    /// Queries surface support for a given queue family index on a physical device.
    ///
    /// # Safety
    /// The `physical_device` handle must be valid and compatible with this surface.
    pub unsafe fn get_physical_device_surface_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> Result<bool> {
        self.surface_loader
            .get_physical_device_surface_support(physical_device, queue_family_index, self.surface)
            .map_err(GfxHalError::VulkanError)
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        tracing::debug!("Destroying Vulkan surface...");
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
        tracing::debug!("Vulkan surface destroyed.");
    }
}
