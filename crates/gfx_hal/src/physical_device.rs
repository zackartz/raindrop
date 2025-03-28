use ash::vk;

use crate::{error::GfxHalError, instance::Instance};

use std::sync::Arc;

/// Represents a physical Vulkan device (GPU).
///
/// This struct holds a handle to the Vulkan physical device and a
/// reference back to the `Instance` it originated from. It does *not* own
/// the `vk::PhysicalDevice` in the sense that it doesn't destroy it; physical
/// devices are implicitly managed by the `vk::Instance`
///
/// It's cheap to clone as it only clones the `Arc<Instance>` and copies the handle
#[derive(Clone)]
pub struct PhysicalDevice {
    /// Shared reference to the Vulkan instance
    instance: Arc<Instance>,
    /// The raw Vulkan physical device handle.
    handle: vk::PhysicalDevice,
}

/// Holds information about queue families found on a `PhysicalDevice`.
#[derive(Debug, Clone, Default)]
pub struct QueueFamilyIndices {
    /// Queue family index supporting graphics operations.
    pub graphics_family: Option<u32>,
    /// Queue family index supporting compute operations.
    pub compute_family: Option<u32>,
    /// Queue family index supporting transfer operations.
    pub transfer_family: Option<u32>,
    /// Queue family index supporting presentaiton to a given surface.
    /// This might be the same as the graphics family.
    pub present_family: Option<u32>,
}

impl QueueFamilyIndices {
    /// Checks if all essential queue families (graphics, present if surface exists) were found.
    pub fn is_complete(&self, requires_present: bool) -> bool {
        self.graphics_family.is_some() && (!requires_present || self.present_family.is_some())
    }
}

/// Represents the suitability of a physical device.
#[derive(Debug)]
pub enum Suitability<'a> {
    /// The device is suitable and meets requirements.
    Suitable {
        /// A score indicating preference (higher is better).
        score: u32,
        /// The indicies of the required queue families.
        indicies: QueueFamilyIndices,
        /// The properties of the device.
        properties: Box<vk::PhysicalDeviceProperties>,
        /// The supported base features of the device.
        features: Box<vk::PhysicalDeviceFeatures>,
        /// THe supported mesh shader features.
        mesh_shader_features: vk::PhysicalDeviceMeshShaderFeaturesEXT<'a>,
    },
    /// The device is not suitable.
    NotSuitable {
        /// The reason why the device is not suitable.
        reason: String,
    },
}

impl PhysicalDevice {
    /// Creates a new `PhysicalDevice` wrapper
    /// Typically called internally by `Instance::enumerate_physical_devices`
    pub(crate) fn new(instance: Arc<Instance>, handle: vk::PhysicalDevice) -> Self {
        Self { instance, handle }
    }

    /// Gets the raw `vk::PhysicalDevice` handle.
    pub fn handle(&self) -> vk::PhysicalDevice {
        self.handle
    }

    /// Gets a reference to the `Instance` this device belongs to.
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Queries the basic properties of the physical device.
    ///
    /// # Safety
    /// Assumes the `PhysicalDevice` handle is valid.
    pub unsafe fn get_properties(&self) -> vk::PhysicalDeviceProperties {
        self.instance
            .ash_instance()
            .get_physical_device_properties(self.handle)
    }

    /// Queries the supported features, including mesh shaders.
    ///
    /// # Safety
    /// Assumes the `PhysicalDevice` handle is valid.
    pub unsafe fn get_features(
        &self,
    ) -> (
        vk::PhysicalDeviceFeatures,
        vk::PhysicalDeviceMeshShaderFeaturesEXT,
        vk::PhysicalDeviceDynamicRenderingFeatures,
    ) {
        let mut mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut mesh_shader_features)
            .push_next(&mut dynamic_rendering_features);

        self.instance
            .ash_instance()
            .get_physical_device_features2(self.handle, &mut features2);

        (
            features2.features,
            mesh_shader_features,
            dynamic_rendering_features,
        )
    }

    /// Queries the properties of all queue families available on the device.
    ///
    /// # Safety
    /// Assumes the `PhysicalDevice` handle is valid.
    pub unsafe fn get_queue_family_properties(&self) -> Vec<vk::QueueFamilyProperties> {
        self.instance
            .ash_instance()
            .get_physical_device_queue_family_properties(self.handle)
    }

    /// Queries the device specific extensions supported by this physical device.
    ///
    /// # Safety
    /// Assumes the `PhysicalDevice` handle is valid.
    pub unsafe fn get_supported_extensions(
        &self,
    ) -> Result<Vec<vk::ExtensionProperties>, GfxHalError> {
        self.instance
            .ash_instance()
            .enumerate_device_extension_properties(self.handle)
            .map_err(GfxHalError::VulkanError)
    }

    // /// Finds suitable queue family indicies based on required flags and optional surface.
    // ///
    // /// # Safety
    // /// Assumes the `PhysicalDevice` handle and `Surface` (if provided) are valid.
    // pub unsafe fn find_queue_families(
    //     &self,
    //     surface: Option<&Surface>,
    // ) -> Result<QueueFamilyIndices, GfxHalError> {
    // }

    // /// Checks if the physical device meets the specified requirements and scores it.
    // ///
    // /// # Arguments
    // /// * `required_extensions` - A slice of C-style strings representing required device extensions (e.g., `ash::extensions::khr::Swapchain::name()`).
    // /// * `required_mesh_features` - The minimum mesh shader features required. Check `task_shader` and `mesh_shader` fields.
    // /// * `surface` - An optional surface to check for presentation support.
    // ///
    // /// # Safety
    // /// Assumes the `PhysicalDevice` handle and `Surface` (if provided) are valid.
    // pub unsafe fn check_suitability(
    //     &self,
    //     required_extensions: &[&CStr],
    //     required_mesh_features: &vk::PhysicalDeviceMeshShaderFeaturesEXT,
    //     surface: Option<&Surface>,
    // ) -> Result<Suitability, GfxHalError> {
    // }
}
