use std::{
    error::Error,
    ffi::{CStr, CString},
    fs::OpenOptions,
    sync::Arc,
    time::Instant,
};

use ash::vk;
use gfx_hal::{
    device::Device, error::GfxHalError, instance::Instance, instance::InstanceConfig,
    physical_device::PhysicalDevice, queue::Queue, surface::Surface,
};
use raw_window_handle::HasDisplayHandle;
use renderer::{Renderer, RendererError};
use resource_manager::{ResourceManager, ResourceManagerError};
use tracing::{debug, error, info, warn};
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

// --- Configuration ---
const WINDOW_TITLE: &str = "Rust Vulkan Egui Engine";
const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 720;
const APP_NAME: &str = "My App";
const ENGINE_NAME: &str = "My Engine";

// --- Error Handling ---
#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("Window Creation Error: {0}")]
    WindowCreation(#[from] winit::error::OsError),
    #[error("Graphics HAL Error: {0}")]
    GfxHal(#[from] GfxHalError),
    #[error("Resource Manager Error: {0}")]
    ResourceManager(#[from] ResourceManagerError),
    #[error("Renderer Error: {0}")]
    Renderer(#[from] RendererError),
    #[error("Suitable physical device not found")]
    NoSuitableDevice,
    #[error("Required queue family not found")]
    NoSuitableQueueFamily,
    #[error("Failed to create CString: {0}")]
    NulError(#[from] std::ffi::NulError),
    #[error("Missing required Vulkan extension: {0}")]
    MissingExtension(String),
}

// --- Main Application Structure ---
struct Application {
    // Core Vulkan Objects (managed by gfx_hal)
    _instance: Arc<Instance>,         // Keep instance alive
    _physical_device: PhysicalDevice, // Keep info, though Device holds handle
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    surface: Arc<Surface>,

    // Resource Management
    resource_manager: Arc<ResourceManager>,

    // Renderer
    renderer: Renderer,

    // Windowing
    window: Arc<Window>, // Use Arc for potential multi-threading later

    // State
    last_frame_time: Instant,
    ui_show_demo: bool,
}

impl Application {
    fn new(window: Arc<Window>) -> Result<Self, AppError> {
        info!("Initializing Application...");

        // --- 1. gfx_hal Setup ---
        let instance_extensions = [
            // Add extensions required by the platform (e.g., from winit)
            // ash::extensions::ext::DebugUtils::name(), // If using validation
            ash::khr::surface::NAME,
            // Platform specific (example for Xlib/Wayland)
            #[cfg(target_os = "linux")]
            ash::khr::xlib_surface::NAME,
            #[cfg(target_os = "linux")]
            ash::khr::wayland_surface::NAME,
            // Add other platform extensions as needed (Win32, Metal, etc.)
        ];
        let instance_extensions_c: Vec<CString> = instance_extensions
            .iter()
            .map(|&s| CString::new(s.to_bytes()).unwrap())
            .collect();

        let instance_config = InstanceConfig {
            application_name: APP_NAME.to_string(),
            engine_name: ENGINE_NAME.to_string(),
            enable_validation: cfg!(debug_assertions), // Enable validation in debug
            ..Default::default()
        };

        let instance = Instance::new(
            &instance_config,
            &window.display_handle().unwrap(),
            &instance_extensions_c, // Pass external extensions
        )?;
        info!("Vulkan Instance created.");

        let surface = unsafe {
            // Need unsafe for create_surface
            instance.create_surface(window.as_ref())? // Pass window ref
        };
        info!("Vulkan Surface created.");

        // --- 2. Physical Device Selection ---
        let required_device_extensions =
            [ash::khr::swapchain::NAME, ash::khr::dynamic_rendering::NAME];
        let required_device_extensions_cstr: Vec<&CStr> = required_device_extensions
            .iter()
            .map(|s| CStr::from_bytes_with_nul(s.to_bytes_with_nul()).unwrap())
            .collect();

        // Define required features (Dynamic Rendering is crucial)
        let required_dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default().dynamic_rendering(true);
        // Chain other required features if necessary (e.g., mesh shader)
        // let mut required_mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::builder()...
        // required_dynamic_rendering_features = required_dynamic_rendering_features.push_next(&mut required_mesh_shader_features);

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let (physical_device, queue_family_indices) = physical_devices
            .into_iter()
            .find_map(|pd| {
                match find_suitable_device_and_queues(
                    &pd,
                    &surface,
                    &required_device_extensions_cstr,
                    &required_dynamic_rendering_features,
                ) {
                    Ok(indices) => Some((pd, indices)),
                    Err(e) => {
                        warn!(
                            "Skipping physical device {:?}: {}",
                            unsafe {
                                instance
                                    .ash_instance()
                                    .get_physical_device_properties(pd.handle())
                                    .device_name_as_c_str()
                            },
                            e
                        );
                        None
                    }
                }
            })
            .ok_or(AppError::NoSuitableDevice)?;

        let pd_props = unsafe {
            instance
                .ash_instance()
                .get_physical_device_properties(physical_device.handle())
        };
        info!(
            "Selected Physical Device: {}",
            pd_props.device_name_as_c_str().unwrap().to_string_lossy()
        );
        debug!("Using Queue Families: {:?}", queue_family_indices);

        // --- 3. Logical Device and Queues ---
        // Enable required features
        let enabled_features = vk::PhysicalDeviceFeatures::default(); // Add base features if needed

        let enabled_buffer_device_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);

        let enabled_dynamic_rendering = required_dynamic_rendering_features; // Copy the builder state

        let device = unsafe {
            // Need unsafe for create_logical_device
            physical_device.create_logical_device(
                &required_device_extensions_cstr,
                &queue_family_indices,
                &enabled_features,
                None,
                &enabled_dynamic_rendering, // Pass features to enable
                &enabled_buffer_device_address,
            )?
        };
        let device_handle_at_creation = device.raw().handle();
        info!(
            "App: Created Device handle: {:?}",
            device_handle_at_creation
        );

        // Get specific queues (assuming graphics and present are the same for simplicity)
        let graphics_queue = device.get_graphics_queue();
        let queue_associated_device_handle = graphics_queue.device().raw().handle();
        info!(
            "App: Queue is associated with Device handle: {:?}",
            queue_associated_device_handle
        );
        assert_eq!(
            device_handle_at_creation, queue_associated_device_handle,
            "Device handle mismatch immediately after queue creation!"
        );

        // --- 4. Resource Manager ---
        let resource_manager = Arc::new(ResourceManager::new(instance.clone(), device.clone())?);
        info!("Resource Manager initialized.");

        let renderer_device_handle_to_pass = device.raw().handle();
        let renderer_queue_device_handle_to_pass = graphics_queue.device().raw().handle();
        info!(
            "App: Passing Device handle to Renderer: {:?}",
            renderer_device_handle_to_pass
        );
        info!(
            "App: Passing Queue associated with Device handle: {:?}",
            renderer_queue_device_handle_to_pass
        );

        // --- 5. Renderer ---
        let initial_size = window.inner_size();
        let renderer = Renderer::new(
            instance.clone(), // Pass instance for allocator creation
            device.clone(),
            graphics_queue.clone(),
            surface.clone(),
            resource_manager.clone(),
            initial_size.width,
            initial_size.height,
        )?;
        info!("Renderer initialized.");

        Ok(Self {
            _instance: instance,
            _physical_device: physical_device,
            device,
            graphics_queue,
            surface,
            resource_manager,
            renderer,
            window,
            last_frame_time: Instant::now(),
            ui_show_demo: true,
        })
    }

    fn handle_event(&mut self, event: &Event<()>, active_event_loop: &ActiveEventLoop) {
        match event {
            Event::WindowEvent { event, window_id } if *window_id == self.window.id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        info!("Close requested. Exiting...");
                        active_event_loop.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        info!(
                            "Window resized to: {}x{}",
                            physical_size.width, physical_size.height
                        );
                        // Important: Resize renderer *before* the next frame
                        self.renderer
                            .resize(physical_size.width, physical_size.height);
                        // Egui also needs the new screen descriptor info, though
                        // egui_winit_state might handle this internally via on_window_event.
                        // Explicitly setting it might be safer depending on version.
                        // self.egui_winit_state.set_max_size_points(...) // If needed
                    }
                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        info!("Scale factor changed: {}", scale_factor);
                        // May also need to resize renderer if size depends on scale factor
                        let new_inner_size = self.window.inner_size();
                        self.renderer
                            .resize(new_inner_size.width, new_inner_size.height);
                    }
                    // Handle other inputs if not consumed by egui
                    WindowEvent::KeyboardInput { .. }
                    | WindowEvent::CursorMoved { .. }
                    | WindowEvent::MouseInput { .. } => {}
                    _ => {}
                }
            }
            // Event::MainEventsCleared => { // Use AboutToWait for newer winit
            //     // Application update code.
            //     self.window.request_redraw();
            // }
            Event::AboutToWait => {
                // Application update code and redraw request.
                // This is the main place to prepare and trigger rendering.

                let now = Instant::now();
                let _delta_time = now.duration_since(self.last_frame_time);
                self.last_frame_time = now;

                // --- Render Frame ---
                match self.renderer.render_frame() {
                    Ok(_) => {}
                    Err(RendererError::SwapchainSuboptimal) => {
                        // Swapchain is suboptimal, recreate it next frame by triggering resize
                        warn!("Swapchain suboptimal, forcing resize.");
                        let size = self.window.inner_size();
                        self.renderer.resize(size.width, size.height);
                    }
                    Err(e) => {
                        error!("Failed to render frame: {}", e);
                        // Decide how to handle persistent errors (e.g., exit)
                        active_event_loop.exit();
                    }
                }
            }
            Event::LoopExiting => {
                info!("Event loop exiting. Cleaning up...");
                // Wait for GPU to finish before dropping resources
                if let Err(e) = self.device.wait_idle() {
                    error!("Error waiting for device idle on exit: {}", e);
                }
                info!("GPU idle. Cleanup complete.");
            }
            _ => {}
        }
    }
}

// --- Helper Functions ---

/// Finds queue family indices for graphics and presentation.
fn find_suitable_device_and_queues(
    physical_device: &PhysicalDevice,
    surface: &Surface,
    required_extensions: &[&CStr],
    required_dynamic_rendering_features: &vk::PhysicalDeviceDynamicRenderingFeaturesKHR,
) -> Result<gfx_hal::physical_device::QueueFamilyIndices, Box<dyn Error>> {
    // 1. Check Extension Support
    let supported_extensions = unsafe {
        physical_device
            .instance()
            .ash_instance()
            .enumerate_device_extension_properties(physical_device.handle())?
    };
    let supported_extension_names: std::collections::HashSet<&CStr> = supported_extensions
        .iter()
        .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
        .collect();

    for &required in required_extensions {
        if !supported_extension_names.contains(required) {
            return Err(
                format!("Missing required extension: {}", required.to_string_lossy()).into(),
            );
        }
    }

    // 2. Check Feature Support (Dynamic Rendering)
    let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default();
    let mut features2 =
        vk::PhysicalDeviceFeatures2::default().push_next(&mut dynamic_rendering_features);

    unsafe {
        physical_device
            .instance()
            .ash_instance()
            .get_physical_device_features2(physical_device.handle(), &mut features2);
    }

    if dynamic_rendering_features.dynamic_rendering == vk::FALSE {
        return Err("Dynamic Rendering feature not supported".into());
    }
    // Add checks for other required features here...

    // 3. Check Queue Family Support
    let queue_family_properties = unsafe {
        physical_device
            .instance()
            .ash_instance()
            .get_physical_device_queue_family_properties(physical_device.handle())
    };

    let mut graphics_family = None;
    let mut present_family = None;

    for (i, queue_family) in queue_family_properties.iter().enumerate() {
        let index = i as u32;

        // Check for graphics support
        if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics_family = Some(index);
        }

        // Check for presentation support
        let present_support = unsafe {
            surface
                .surface_loader()
                .get_physical_device_surface_support(
                    physical_device.handle(),
                    index,
                    surface.handle(),
                )?
        };
        if present_support {
            present_family = Some(index);
        }

        if graphics_family.is_some() && present_family.is_some() {
            break; // Found suitable families
        }
    }

    match (graphics_family, present_family) {
        (Some(graphics), Some(present)) => Ok(gfx_hal::physical_device::QueueFamilyIndices {
            graphics_family: Some(graphics),
            present_family: Some(present), // Could be the same as graphics
            compute_family: None,          // Not needed for this example
            transfer_family: None,         // Not needed for this example
        }),
        _ => Err("Could not find suitable queue families".into()),
    }
}

// --- Entry Point ---
fn main() -> Result<(), Box<dyn Error>> {
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(true)
        .with_file(false)
        .with_line_number(false)
        .without_time();

    let log_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("log-debug.log")?;

    let json_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(log_file)
        .with_filter(filter::LevelFilter::TRACE);

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(json_layer)
        .init();

    // --- Winit Setup ---
    let event_loop = EventLoop::new()?;
    let window = Arc::new(event_loop.create_window(WindowAttributes::default())?);

    info!("Window created.");

    // --- Application Setup ---
    let mut app = Application::new(window.clone())?;

    // --- Event Loop ---
    info!("Starting event loop...");
    event_loop.run(move |event, elwt| {
        // elwt is EventLoopWindowTarget, not needed directly here often
        app.handle_event(&event, elwt);
    })?;

    Ok(())
}
