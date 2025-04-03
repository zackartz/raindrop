use ash::vk;
use std::collections::HashSet;
use std::ffi::CStr;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::error::{GfxHalError, Result};
use crate::instance::Instance;
use crate::physical_device::{PhysicalDevice, QueueFamilyIndices};
use crate::queue::Queue;

/// Represents the logical Vulkan device, created from a `PhysicalDevice`.
///
/// Owns the `ash::Device` and provides access to device functions and queues.
pub struct Device {
    _instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queues: Mutex<HashMap<(u32, u32), Arc<Queue>>>,
    graphics_queue_family_index: u32,
    compute_queue_family_index: Option<u32>,
    transfer_queue_family_index: Option<u32>,
}

impl Device {
    /// Creates a new logical device. Typically called via `PhysicalDevice::create_logical_device`.
    /// Uses a two-stage initialization to avoid Arc::new_cyclic issues.
    ///
    /// # Safety
    /// - `instance` and `physical_device_handle` must be valid.
    /// - `queue_family_indicies` must be valid indicies obtained from the `physical_device_handle`.
    /// - `required_extensions` must be supported by the `physical_device_handle`.
    /// - All feature structs passed must be supported by the `physical_device_handle`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new(
        instance: Arc<Instance>,
        physical_device_handle: vk::PhysicalDevice,
        queue_family_indicies: &QueueFamilyIndices,
        required_extensions: &[&CStr],
        enabled_features: &vk::PhysicalDeviceFeatures,
        mesh_features: Option<&vk::PhysicalDeviceMeshShaderFeaturesEXT>,
        dynamic_rendering_features: &vk::PhysicalDeviceDynamicRenderingFeatures,
        buffer_device_address_features: &vk::PhysicalDeviceBufferDeviceAddressFeatures,
        // Add other feature structs here as needed...
    ) -> Result<Arc<Self>> {
        // --- 1. Prepare Queue Create Infos (Same as before) ---
        let mut queue_create_infos = Vec::new();
        let mut unique_queue_families = HashSet::new();
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
        if let Some(present_family) = queue_family_indicies.present_family {
            unique_queue_families.insert(present_family);
        }
        let queue_priorities = [1.0f32];
        for &family_index in &unique_queue_families {
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family_index)
                .queue_priorities(&queue_priorities);
            queue_create_infos.push(queue_create_info);
        }

        // --- 2. Prepare Feature Chain (Same as before) ---
        let extension_names_raw: Vec<*const i8> =
            required_extensions.iter().map(|s| s.as_ptr()).collect();
        let mut features2 = vk::PhysicalDeviceFeatures2::default().features(*enabled_features);
        let mut mesh_features_copy;
        if let Some(mesh_feats) = mesh_features {
            mesh_features_copy = *mesh_feats;
            features2 = features2.push_next(&mut mesh_features_copy);
        }
        let mut dyn_rendering_feats_copy = *dynamic_rendering_features;
        if dyn_rendering_feats_copy.dynamic_rendering != vk::TRUE {
            return Err(GfxHalError::MissingFeature("Dynamic Rendering".to_string()));
        }
        features2 = features2.push_next(&mut dyn_rendering_feats_copy);
        let mut bda_features_copy = *buffer_device_address_features;
        if bda_features_copy.buffer_device_address != vk::TRUE {
            return Err(GfxHalError::MissingFeature(
                "Buffer Device Address".to_string(),
            ));
        }
        features2 = features2.push_next(&mut bda_features_copy);
        // Chain other features here...

        // --- 3. Create the SINGLE ash::Device (Same as before) ---
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names_raw)
            .push_next(&mut features2);
        tracing::info!(
            "Creating logical device with extensions: {:?}",
            required_extensions
        );
        let ash_device = instance.ash_instance().create_device(
            physical_device_handle,
            &device_create_info,
            None,
        )?;
        tracing::info!(
            "Logical device created successfully (ash::Device handle: {:?}).",
            ash_device.handle()
        );

        // --- 4. Create the Device struct in an Arc (Stage 1) ---
        // Initialize the queues map as empty for now.
        let device_arc = Arc::new(Device {
            _instance: instance.clone(),
            physical_device: physical_device_handle,
            device: ash_device, // Move the created ash::Device here
            queues: Mutex::new(HashMap::new()), // Start with empty map
            graphics_queue_family_index: graphics_family,
            compute_queue_family_index: queue_family_indicies.compute_family,
            transfer_queue_family_index: queue_family_indicies.transfer_family,
        });
        tracing::debug!(
            "Device Arc created (Stage 1) with ash::Device handle: {:?}",
            device_arc.raw().handle()
        );

        // --- 5. Create Queues and Populate Map (Stage 2) ---
        // Now that we have the final Arc<Device>, we can create the Queues.
        let mut queues_to_insert = HashMap::new();
        for &family_index in &unique_queue_families {
            // Get the Vulkan queue handle using the device stored in the Arc
            // Assuming queue index 0 for simplicity
            let vk_queue_handle = device_arc.device.get_device_queue(family_index, 0);

            // Create the Queue wrapper, passing a clone of the device_arc
            let queue_wrapper = Arc::new(Queue::new(
                device_arc.clone(), // Pass the Arc<Device>
                vk_queue_handle,
                family_index,
            ));
            queues_to_insert.insert((family_index, 0), queue_wrapper);
            tracing::trace!("Created queue wrapper for family {}", family_index);
        }

        // Lock the mutex and insert the created queues into the map within the Arc<Device>
        {
            // Scope for the mutex guard
            let mut queues_map_guard = device_arc.queues.lock()?;
            *queues_map_guard = queues_to_insert; // Replace the empty map with the populated one
            tracing::debug!(
                "Device Arc populated with {} queues (Stage 2).",
                queues_map_guard.len()
            );
        } // Mutex guard is dropped here

        Ok(device_arc) // Return the fully initialized Arc<Device>
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
    pub fn get_queue(&self, family_index: u32, queue_index: u32) -> Result<Arc<Queue>> {
        if queue_index != 0 {
            tracing::warn!("get_queue currently only supports queue_index 0");
            return Err(GfxHalError::MissingQueueFamily(
                "get_queue only supports queue_index 0".to_string(),
            ));
        }

        self.queues
            .lock()?
            .get(&(family_index, queue_index))
            .cloned()
            .ok_or(GfxHalError::MissingQueueFamily(
                "could not get queue family".to_string(),
            ))
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
        dynamic_rendering_features: &vk::PhysicalDeviceDynamicRenderingFeatures,
        buffer_device_address_features: &vk::PhysicalDeviceBufferDeviceAddressFeatures,
    ) -> Result<Arc<Device>> {
        Device::new(
            Arc::clone(self.instance()),
            self.handle(),
            queue_family_indices,
            required_extensions,
            enabled_features,
            mesh_features,
            dynamic_rendering_features,
            buffer_device_address_features,
        )
    }
}
