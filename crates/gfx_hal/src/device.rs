use ash::vk;
use parking_lot::Mutex;
use std::ffi::CStr;
use std::{collections::HashMap, sync::Arc};

use crate::error::{GfxHalError, Result};
use crate::instance::Instance;
use crate::physical_device::{PhysicalDevice, QueueFamilyIndices};
use crate::queue::Queue;

/// Represents the logical Vulkan device, created from a `PhysicalDevice`.
///
/// Owns the `ash::Device` and provides access to device functions and queues.
pub struct Device {
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queues: Mutex<HashMap<(u32, u32), Arc<Queue>>>,
    graphics_queue_family_index: u32,
    compute_queue_family_index: Option<u32>,
    transfer_queue_family_index: Option<u32>,
}

impl Device {
    /// Creates a new logical device. Typically called via `PhysicalDevice::create_logical_device`.
    ///
    /// # Saftey
    /// - `instance` and `physical_device_handle` must be valid.
    /// - `queue_family_indicies` must be valid indicies obtained from the `physical_device_handle`.
    /// - `required_extensions` must be supported by the `physical_device_handle`.
    /// - `enabled_features` and `mesh_features` must be supported by the `physical_device_handle`.
    pub(crate) unsafe fn new(
        instance: Arc<Instance>,
        physical_device_handle: vk::PhysicalDevice,
        queue_family_indicies: &QueueFamilyIndices,
        required_extensions: &[&CStr],
        enabled_features: &vk::PhysicalDeviceFeatures,
        mesh_features: Option<&vk::PhysicalDeviceMeshShaderFeaturesEXT>,
    ) -> Result<Arc<Self>> {
        let mut queue_create_infos = Vec::new();
        let mut unique_queue_families = std::collections::HashSet::new();

        let graphics_family = queue_family_indicies.graphics_family.ok_or_else(|| {
            GfxHalError::MissingQueueFamily("Graphics Queue Family Missing".to_string())
        })?;
        unique_queue_families.insert(graphics_family);

        if let Some(compute_family) = queue_family_indicies.compute_family {
            unique_queue_families.insert(compute_family);
        }
        if let Some(transfer_family) = queue_family_indicies.transfer_family {
            unique_queue_families.insert(transfer_family);
        }

        let queue_priorities = [1.0f32];
        for &family_index in &unique_queue_families {
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family_index)
                .queue_priorities(&queue_priorities);
            queue_create_infos.push(queue_create_info);
        }

        let extension_names_raw: Vec<*const i8> =
            required_extensions.iter().map(|s| s.as_ptr()).collect();

        let mut features2 = vk::PhysicalDeviceFeatures2::default().features(*enabled_features);
        let mut mesh_features_copy;

        if let Some(mesh_feats) = mesh_features {
            mesh_features_copy = *mesh_feats;
            features2 = features2.push_next(&mut mesh_features_copy);
        }

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names_raw)
            .push_next(&mut features2);

        tracing::info!(
            "Creating logical device with extensions: {:?}",
            required_extensions
        );
        let device = instance.ash_instance().create_device(
            physical_device_handle,
            &device_create_info,
            None,
        )?;
        tracing::info!("logical device created successfully.");

        let mut queues_map = HashMap::new();
        let arc_device_placeholder = Arc::new(Self {
            instance,
            physical_device: physical_device_handle,
            device,
            queues: Mutex::new(HashMap::new()),
            graphics_queue_family_index: graphics_family,
            compute_queue_family_index: queue_family_indicies.compute_family,
            transfer_queue_family_index: queue_family_indicies.transfer_family,
        });

        for &family_index in &unique_queue_families {
            let queue_handler = arc_device_placeholder
                .device
                .get_device_queue(family_index, 0);
            let queue_wrapper = Arc::new(Queue::new(
                Arc::clone(&arc_device_placeholder),
                queue_handler,
                family_index,
            ));
            queues_map.insert((family_index, 0), queue_wrapper);
        }

        let device_handle = unsafe {
            arc_device_placeholder
                .instance
                .ash_instance()
                .create_device(physical_device_handle, &device_create_info, None)?
        };

        let final_device = Arc::new(Self {
            instance: Arc::clone(&arc_device_placeholder.instance), // Clone from placeholder
            physical_device: physical_device_handle,
            device: device_handle,          // Use the newly created handle
            queues: Mutex::new(queues_map), // Use the populated map
            graphics_queue_family_index: graphics_family,
            compute_queue_family_index: queue_family_indicies.compute_family,
            transfer_queue_family_index: queue_family_indicies.transfer_family,
        });

        Ok(final_device)
    }

    /// Provides raw access to the underlying `ash::Device`.
    /// Use with caution, prefer safe wrappers where possible.
    pub fn raw(&self) -> &ash::Device {
        &self.device
    }

    /// Gets the handle of the physical device this logical device was created from.
    pub fn physical_device_handle(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Gets the primary graphics queue family index used by this device.
    pub fn graphics_queue_family_index(&self) -> u32 {
        self.graphics_queue_family_index
    }

    /// Gets the compute queue family index, if a distinct one was found/used.
    pub fn compute_queue_family_index(&self) -> Option<u32> {
        self.compute_queue_family_index
    }

    /// Gets the transfer queue family index, if a distinct one was found/used.
    pub fn transfer_queue_family_index(&self) -> Option<u32> {
        self.transfer_queue_family_index
    }

    /// Gets a wrapped queue handle.
    /// Currently only supports queue index 0 for each family.
    pub fn get_queue(&self, family_index: u32, queue_index: u32) -> Option<Arc<Queue>> {
        if queue_index != 0 {
            tracing::warn!("get_queue currently only supports queue_index 0");
            return None;
        }
        self.queues
            .lock()
            .get(&(family_index, queue_index))
            .cloned()
    }

    /// Gets the primary graphics queue (family index from `graphics_queue_family_index`, queue index 0).
    /// Panics if the graphics queue wasn't successfully created.
    pub fn get_graphics_queue(&self) -> Arc<Queue> {
        self.get_queue(self.graphics_queue_family_index, 0)
            .expect("Graphics queue should always exist")
    }

    /// Waits until the logical device becomes idle.
    /// This is a heavy operation and should be used sparingly (e.g., before destruction).
    pub fn wait_idle(&self) -> Result<()> {
        tracing::debug!("Waiting for device idle...");
        unsafe { self.device.device_wait_idle()? };
        tracing::debug!("Device idle.");
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        tracing::debug!("Destroying logical device...");
        if let Err(e) = self.wait_idle() {
            tracing::error!("Error waiting for device idle during drop: {}", e);
        }
        unsafe {
            self.device.destroy_device(None);
        }
        tracing::debug!("Logical device destroyed.");
    }
}

impl PhysicalDevice {
    /// Creates the logical device (`Device`) from this physical device.
    ///
    /// # Safety
    /// See `Device::new` safety comments.
    pub unsafe fn create_logical_device(
        &self,
        required_extensions: &[&CStr],
        queue_family_indices: &QueueFamilyIndices,
        enabled_features: &vk::PhysicalDeviceFeatures,
        mesh_features: Option<&vk::PhysicalDeviceMeshShaderFeaturesEXT>,
    ) -> Result<Arc<Device>> {
        Device::new(
            Arc::clone(self.instance()),
            self.handle(),
            queue_family_indices,
            required_extensions,
            enabled_features,
            mesh_features,
        )
    }
}
