use std::{
    collections::HashSet,
    ffi::{c_char, c_void, CStr, CString},
    sync::Arc,
};

use ash::{ext::debug_utils, vk};
use winit::raw_window_handle::{DisplayHandle, HasDisplayHandle};

use crate::error::{GfxHalError, Result};

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number;
    let message_id_name = if callback_data.p_message_id_name.is_null() {
        std::borrow::Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };
    let message = if callback_data.p_message.is_null() {
        std::borrow::Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[VERBOSE]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[INFO]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[WARNING]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[ERROR]",
        _ => "[UNKNOWN SEVERITY]",
    };
    let ty = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[GENERAL]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[VALIDATION]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[PERFORMANCE]",
        _ => "[UNKNOWN TYPE]",
    };

    // Use the tracing crate for output
    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            tracing::debug!(
                "{} {} ({}:{}) {}",
                severity,
                ty,
                message_id_name,
                message_id_number,
                message
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            tracing::info!(
                "{} {} ({}:{}) {}",
                severity,
                ty,
                message_id_name,
                message_id_number,
                message
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            tracing::warn!(
                "{} {} ({}:{}) {}",
                severity,
                ty,
                message_id_name,
                message_id_number,
                message
            );
        }
        // Treat ERROR and higher as errors
        _ => {
            tracing::error!(
                "{} {} ({}:{}) {}",
                severity,
                ty,
                message_id_name,
                message_id_number,
                message
            );
        }
    }

    vk::FALSE // Standard return value
}

#[derive(Clone, Debug)]
pub struct InstanceConfig {
    pub application_name: String,
    pub engine_name: String,
    pub application_version: u32,
    pub engine_version: u32,
    /// Enable Vulkan validation layers
    pub enable_validation: bool,
    /// Additional required instance extensions beyond surface/debug.
    pub required_extensions: Vec<&'static CStr>,
}

impl Default for InstanceConfig {
    fn default() -> Self {
        InstanceConfig {
            application_name: "Defualt App".to_string(),
            engine_name: "Default Engine".to_string(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            enable_validation: cfg!(debug_assertions),
            required_extensions: Vec::new(),
        }
    }
}

/// Represents the Vulkan API Instance
///
/// Owns the `ash::Entry`, `ash::Instance` and potentially the debug messenger.
/// This is the starting point for interacting with Vulkan
pub struct Instance {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl Instance {
    /// Creates a new Vulkan `Instance`
    ///
    /// # Arguments
    /// * `config` - Configuration settings for the instance
    /// * `display_handle` - Raw display handle for the windowing system
    /// * `external_extensions` - A slice of `CString` representing additional required instance extensions,
    ///                           typically provided by integration libraries like `egui-ash`
    pub fn new(
        config: &InstanceConfig,
        display_handle: &dyn HasDisplayHandle,
        external_extentions: &[CString],
    ) -> Result<Arc<Self>> {
        let entry = unsafe { ash::Entry::load()? };

        let app_name = CString::new(config.application_name.clone())?;
        let engine_name = CString::new(config.engine_name.clone())?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(config.application_version)
            .engine_name(&engine_name)
            .engine_version(config.engine_version)
            .api_version(vk::API_VERSION_1_3);

        let validation_layers = [c"VK_LAYER_KHRONOS_validation"];
        let enabled_layer_names_raw: Vec<*const c_char> = if config.enable_validation
            && Self::check_validation_layer_support(&entry, &validation_layers)?
        {
            tracing::info!("Validation layers enabled.");
            validation_layers.iter().map(|name| name.as_ptr()).collect()
        } else {
            if config.enable_validation {
                tracing::warn!("Validation layers requested but not supported. Disabling.");
            }
            Vec::new()
        };

        let display_handle: DisplayHandle = display_handle.display_handle()?;

        let surface_extensions_ptrs =
            ash_window::enumerate_required_extensions(display_handle.into())?;

        let mut required_cstrs_for_check: Vec<&CStr> = Vec::new();

        let surface_extensions_cstrs: Vec<&CStr> = surface_extensions_ptrs
            .iter()
            .map(|&ptr| unsafe { CStr::from_ptr(ptr) })
            .collect();
        required_cstrs_for_check.extend(&surface_extensions_cstrs);

        required_cstrs_for_check.extend(external_extentions.iter().map(|cs| cs.as_c_str()));

        if config.enable_validation {
            required_cstrs_for_check.push(ash::ext::debug_utils::NAME);
        }

        required_cstrs_for_check.sort_unstable();
        required_cstrs_for_check.dedup();

        Self::check_instance_extension_support(&entry, &required_cstrs_for_check)?;
        tracing::info!(
            "Required instance extensions supported: {:?}",
            required_cstrs_for_check
        );

        let mut enabled_extension_names_raw: Vec<*const c_char> = Vec::new();
        enabled_extension_names_raw.extend(surface_extensions_ptrs);
        enabled_extension_names_raw.extend(external_extentions.iter().map(|cs| cs.as_ptr()));
        if config.enable_validation {
            enabled_extension_names_raw.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        enabled_extension_names_raw.sort_unstable();
        enabled_extension_names_raw.dedup();

        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&enabled_layer_names_raw)
            .enabled_extension_names(&enabled_extension_names_raw);

        if config.enable_validation {
            instance_create_info = instance_create_info.push_next(&mut debug_create_info);
        }

        let instance = unsafe { entry.create_instance(&instance_create_info, None)? };
        tracing::info!("Vulkan instance created succesfully.");

        let (debug_utils, debug_messenger) = if config.enable_validation {
            let utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messenger =
                unsafe { utils.create_debug_utils_messenger(&debug_create_info, None)? };
            tracing::debug!("Debug messenger created.");
            (Some(utils), Some(messenger))
        } else {
            (None, None)
        };

        Ok(Arc::new(Self {
            entry,
            instance,
            debug_utils,
            debug_messenger,
        }))
    }

    /// Provides access to the loaded Vulkan entry points.
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    /// Provides access to the raw `ash::Instance`.
    pub fn ash_instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Provides access to the DebugUtils extension loader, if enabled.
    pub fn debug_utils(&self) -> Option<&debug_utils::Instance> {
        self.debug_utils.as_ref()
    }

    /// Checks if the requested validation layers are available.
    fn check_validation_layer_support(
        entry: &ash::Entry,
        required_layers: &[&CStr],
    ) -> Result<bool> {
        let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };
        let available_names: HashSet<&CStr> = available_layers
            .iter()
            .map(|layer| unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) })
            .collect();

        for layer in required_layers {
            if !available_names.contains(layer) {
                tracing::warn!("Required validation layer {:?} not found.", layer);
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Checks if the requested instance extensions are available.
    /// Takes a deduplicated list of required extension names (&CStr).
    fn check_instance_extension_support(
        entry: &ash::Entry,
        required_extensions: &[&CStr],
    ) -> Result<()> {
        let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None)? };
        let available_names: HashSet<&CStr> = available_extensions
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
            .collect();
        tracing::debug!("Available instance extensions: {:?}", available_names);

        for ext in required_extensions {
            if !available_names.contains(ext) {
                tracing::error!("Missing required instance extension: {:?}", ext);
                return Err(GfxHalError::MissingExtension(
                    ext.to_string_lossy().into_owned(),
                ));
            }
        }
        Ok(())
    }

    // /// Enumerates all physical devices available to this instance.
    // ///
    // /// # Safety
    // /// The `Instance` must be kept alive while the returned `PhysicalDevices`s are in use.
    // /// This is ensured by returning `PhysicalDevice`s holding an `Arc<Instance>`.
    // pub unsafe fn enumerate_phyiscal_devices(self: &Arc<Self>) -> Result<Vec<PhysicalDevice>> {
    //     let physical_device_handles = self.instance.enumerate_physical_devices()?;
    //
    //     if physical_device_handles.is_empty() {
    //         return Err(GfxHalError::NoSuitableGpu(
    //             "No Vulkan-compatibile GPUs found.".to_string(),
    //         ));
    //     }
    //
    //     let devices = physical_device_handles
    //         .into_iter()
    //         .map(|handle| PhysicalDevice::new(Arc::clone(self), handle))
    //         .collect()?;
    //
    //     Ok(devices)
    // }

    // /// Creates a vulkan surface for the given window
    // ///
    // /// # Safety
    // /// The `window_handle_trait_obj` must point to a valid window/display managed by caller.
    // /// The `Instance` must be kept alive longer than the returned `Surface`
    // pub unsafe fn create_surface(
    //     self: &Arc<Self>,
    //     window_handle_trait_obj: &(impl HasWindowHandle + HasDisplayHandle),
    // ) -> Result<Arc<Surface>> {
    //     Surface::new(Arc::clone(self), window_handle_trait_obj)
    // }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            if let (Some(utils), Some(messenger)) = (&self.debug_utils, self.debug_messenger) {
                tracing::debug!("Destroying debug messenger...");
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            tracing::debug!("Destroying Vulkan instance...");
            self.instance.destroy_instance(None);
            tracing::debug!("Vulkan instance destroyed");
        }
    }
}
