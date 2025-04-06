use std::{
    error::Error,
    ffi::{c_char, CStr, CString},
    fs::OpenOptions,
    sync::Arc,
    time::{Duration, Instant},
};

use ash::vk;
use clap::Parser;
use egui::{Context, Slider, ViewportId};
use egui_winit::State;
use gfx_hal::{
    device::Device, error::GfxHalError, instance::Instance, instance::InstanceConfig,
    physical_device::PhysicalDevice, queue::Queue, surface::Surface,
};
use glam::Vec3;
use raw_window_handle::HasDisplayHandle;
use renderer::{Renderer, RendererError};
use resource_manager::{ResourceManager, ResourceManagerError};
use scene::Scene;
use shared::CameraInfo;
use tracing::{debug, error, info, warn};
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

// --- Configuration ---
const APP_NAME: &str = "BeginDisregard";
const ENGINE_NAME: &str = "Engine";

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
    #[error("Failed to create CString: {0}")]
    NulError(#[from] std::ffi::NulError),
    #[error("Scene Error: {0}")]
    SceneError(#[from] scene::SceneError),
}

struct Application {
    _instance: Arc<Instance>,         // Keep instance alive
    _physical_device: PhysicalDevice, // Keep info, though Device holds handle
    _device: Arc<Device>,
    _graphics_queue: Arc<Queue>,
    _surface: Arc<Surface>,

    // Resource Management
    _resource_manager: Arc<ResourceManager>,

    // Renderer
    renderer: Renderer,

    egui_ctx: Context,
    egui_winit: State,
    egui_app: EditorUI,

    // --- Camera State ---
    camera_info: CameraInfo,
    camera_speed: f32,
    camera_sensitivity: f32,

    // --- Input State ---
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,   // Optional: For flying up
    is_down_pressed: bool, // Optional: For flying down
    is_rmb_pressed: bool,  // Right mouse button
    last_mouse_pos: Option<(f64, f64)>,
    mouse_delta: (f64, f64),
    capture_mouse: bool, // Flag to indicate if mouse should control camera

    // Windowing
    window: Arc<Window>, // Use Arc for potential multi-threading later

    frame_count: u32,
    last_fps_update_time: Instant,
    last_frame_time: Instant,
    current_fps: f64,
}

#[derive(Default)]
struct EditorUI {}

impl EditorUI {
    fn title() -> String {
        "engine".to_string()
    }

    fn build_ui(&mut self, ctx: &egui::Context, current_fps: f64, camera_info: &mut CameraInfo) {
        egui::SidePanel::new(egui::panel::Side::Left, Self::title()).show(ctx, |ui| {
            ui.label(format!("FPS - {:.2}", current_fps));

            ui.separator();

            egui::Grid::new("main_grid")
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("FOV");
                    // Modify the passed-in camera_info
                    ui.add(Slider::new(&mut camera_info.camera_fov, 10.0..=120.0));
                    ui.end_row(); // Good practice in grids

                    // You could add more camera controls here if needed
                    // e.g., sliders for position, target (though direct manipulation is better)
                    ui.label("Camera Pos");
                    ui.label(format!(
                        "({:.1}, {:.1}, {:.1})",
                        camera_info.camera_pos.x,
                        camera_info.camera_pos.y,
                        camera_info.camera_pos.z
                    ));
                    ui.end_row();

                    ui.label("Camera Target");
                    ui.label(format!(
                        "({:.1}, {:.1}, {:.1})",
                        camera_info.camera_target.x,
                        camera_info.camera_target.y,
                        camera_info.camera_target.z
                    ));
                    ui.end_row();
                });

            ui.separator();
            ui.label("Controls:");
            ui.label("RMB + Drag: Look");
            ui.label("WASD: Move");
            ui.label("Space: Up");
            ui.label("Shift: Down");
            ui.label("Hold RMB to activate controls.");
        });

        // let mut tree = create_tree();
        //
        // egui::panel::SidePanel::new(egui::panel::Side::Left, Id::new("main_panel")).show(
        //     ctx,
        //     |ui| {
        //         let mut behavior = TreeBehavior {};
        //         tree.ui(&mut behavior, ui);
        //     },
        // );
    }
}

fn create_tree() -> egui_tiles::Tree<EditorUI> {
    let mut next_view_nr = 0;
    let mut gen_pane = || {
        let pane = EditorUI {};
        next_view_nr += 1;
        pane
    };

    let mut tiles = egui_tiles::Tiles::default();

    let mut tabs = vec![];
    tabs.push({
        let children = (0..7).map(|_| tiles.insert_pane(gen_pane())).collect();
        tiles.insert_horizontal_tile(children)
    });
    tabs.push({
        let cells = (0..11).map(|_| tiles.insert_pane(gen_pane())).collect();
        tiles.insert_grid_tile(cells)
    });
    tabs.push(tiles.insert_pane(gen_pane()));

    let root = tiles.insert_tab_tile(tabs);

    egui_tiles::Tree::new("my_tree", root, tiles)
}

#[derive(Default)]
struct ApplicationWrapper {
    app: Option<Application>,
}

impl ApplicationHandler for ApplicationWrapper {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title(format!("{} - {}", ENGINE_NAME, APP_NAME,)),
                )
                .expect("Windows to be able to be created"),
        );
        self.app = Some(Application::new(window).expect("Unable to create the Application."));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(app) = &mut self.app {
            app.handle_event(&event, event_loop);
        }
    }
}

impl Application {
    fn new(window: Arc<Window>) -> Result<Self, AppError> {
        info!("Initializing Application...");

        // --- 1. gfx_hal Setup ---
        let instance_extensions =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().into())
                .unwrap();

        let instance_extensions_c: Vec<CString> = instance_extensions
            .iter()
            .map(|&ptr| {
                // Safety: We are trusting that the pointers returned by
                // ash_window::enumerate_required_extensions are valid, non-null,
                // null-terminated C strings. This is a standard assumption when
                // working with C APIs via FFI.
                unsafe {
                    // 1. Create a borrowed CStr reference from the raw pointer.
                    let c_str = CStr::from_ptr(ptr as *const c_char); // Cast is optional but common

                    // 2. Convert the borrowed CStr into an owned CString.
                    c_str.to_owned()
                }
            })
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

        // --- 4. Resource Manager ---
        let resource_manager = Arc::new(ResourceManager::new(instance.clone(), device.clone())?);
        info!("Resource Manager initialized.");

        let scene = Scene::from_gltf(
            "./sponza/NewSponza_Main_glTF_003.gltf",
            resource_manager.clone(),
        )?;

        // --- 5. Renderer ---
        let initial_size = window.inner_size();
        let renderer = Renderer::new(
            instance.clone(), // Pass instance for allocator creation
            device.clone(),
            graphics_queue.clone(),
            surface.clone(),
            resource_manager.clone(),
            scene,
            initial_size.width,
            initial_size.height,
        )?;

        let egui_ctx = Context::default();
        let egui_winit = State::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );
        let egui_app = EditorUI::default();

        info!("Renderer initialized.");

        let camera_info = CameraInfo::default(); // Get default camera settings

        Ok(Self {
            _instance: instance,
            _physical_device: physical_device,
            _device: device,
            _graphics_queue: graphics_queue,
            _surface: surface,
            _resource_manager: resource_manager,
            renderer,
            window,
            egui_winit,
            egui_ctx,
            egui_app,

            // --- Camera ---
            camera_info,               // Store the camera state here
            camera_speed: 5.0,         // Adjust as needed
            camera_sensitivity: 0.002, // Adjust as needed

            // --- Input ---
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            is_rmb_pressed: false,
            last_mouse_pos: None,
            mouse_delta: (0.0, 0.0),
            capture_mouse: false, // Start with mouse free
            frame_count: 0,
            current_fps: 0.,
            last_fps_update_time: Instant::now(),
            last_frame_time: Instant::now(),
        })
    }

    fn handle_event(&mut self, event: &WindowEvent, active_event_loop: &ActiveEventLoop) {
        // Let egui process the event first
        let egui_consumed_event = self.egui_winit.on_window_event(&self.window, event);

        // Only process input for camera if egui didn't consume it AND we are capturing
        let process_camera_input = !egui_consumed_event.consumed && self.capture_mouse;

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
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                info!("Scale factor changed: {}", scale_factor);
                // May also need to resize renderer if size depends on scale factor
                let new_inner_size = self.window.inner_size();
                self.renderer
                    .resize(new_inner_size.width, new_inner_size.height);
            }
            // Handle other inputs if not consumed by egui
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Right {
                    let is_pressed = *state == ElementState::Pressed;
                    self.is_rmb_pressed = is_pressed;

                    // Decide whether to capture/release mouse based on RMB
                    // Only capture if pressed *outside* an egui interactive area
                    if is_pressed && !self.egui_ctx.is_pointer_over_area() {
                        self.capture_mouse = true;
                        self.window
                            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                            .or_else(|_| {
                                self.window
                                    .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                            })
                            .unwrap_or(());
                        self.window.set_cursor_visible(false);
                        self.last_mouse_pos = None; // Reset last pos on capture start
                    } else if !is_pressed {
                        self.capture_mouse = false;
                        self.window
                            .set_cursor_grab(winit::window::CursorGrabMode::None)
                            .unwrap_or(());
                        self.window.set_cursor_visible(true);
                        self.mouse_delta = (0.0, 0.0); // Stop camera movement
                    }
                }
                // Let egui handle its mouse clicks regardless of capture state
                // (handled by on_window_event)
            }

            WindowEvent::CursorMoved { position, .. } => {
                let current_pos = (position.x, position.y);
                if self.capture_mouse {
                    // Only calculate delta if capturing
                    if let Some(last_pos) = self.last_mouse_pos {
                        self.mouse_delta.0 += current_pos.0 - last_pos.0;
                        self.mouse_delta.1 += current_pos.1 - last_pos.1;
                    }
                    // Store position relative to window center might be more robust
                    // with set_cursor_position, but this works with grab/confine too.
                    self.last_mouse_pos = Some(current_pos);
                } else {
                    // Still update egui's pointer position even if not capturing
                    // (handled by on_window_event)
                    self.last_mouse_pos = None; // Reset if not capturing
                }
            }

            // Use PhysicalKey for layout-independent keys
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        ..
                    },
                ..
            } => {
                // Let egui handle keyboard input first if it wants it
                if egui_consumed_event.consumed {
                    return;
                }

                let is_pressed = *state == ElementState::Pressed;
                match physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                        self.is_forward_pressed = is_pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                        self.is_backward_pressed = is_pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                        self.is_left_pressed = is_pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                        self.is_right_pressed = is_pressed;
                    }
                    PhysicalKey::Code(KeyCode::Space) => {
                        self.is_up_pressed = is_pressed;
                    }
                    PhysicalKey::Code(KeyCode::ShiftLeft)
                    | PhysicalKey::Code(KeyCode::ShiftRight) => {
                        self.is_down_pressed = is_pressed;
                    }
                    // Optional: Escape to release mouse capture
                    PhysicalKey::Code(KeyCode::Escape) if is_pressed && self.capture_mouse => {
                        self.capture_mouse = false;
                        self.is_rmb_pressed = false; // Ensure RMB state is also reset
                        self.window
                            .set_cursor_grab(winit::window::CursorGrabMode::None)
                            .unwrap_or(());
                        self.window.set_cursor_visible(true);
                        self.mouse_delta = (0.0, 0.0);
                    }
                    _ => {}
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let delta_time = now.duration_since(self.last_frame_time).as_secs_f32();
                self.last_frame_time = now;

                // --- FPS Calculation ---
                let elapsed_since_last_update = now.duration_since(self.last_fps_update_time);
                self.frame_count += 1;
                if elapsed_since_last_update >= Duration::from_secs(1) {
                    self.current_fps =
                        self.frame_count as f64 / elapsed_since_last_update.as_secs_f64();
                    let new_title = format!(
                        "{} - {} - {:.0} FPS",
                        ENGINE_NAME, APP_NAME, self.current_fps
                    );
                    self.window.set_title(&new_title);
                    self.frame_count = 0;
                    self.last_fps_update_time = now;
                }

                self.update_camera(delta_time); // Call the new update function

                let raw_input = self.egui_winit.take_egui_input(&self.window);

                let egui::FullOutput {
                    platform_output,
                    textures_delta,
                    shapes,
                    pixels_per_point,
                    ..
                } = self.egui_ctx.run(raw_input, |ctx| {
                    self.egui_app
                        .build_ui(ctx, self.current_fps, &mut self.camera_info);
                });

                self.renderer.update_textures(textures_delta).unwrap();

                self.egui_winit
                    .handle_platform_output(&self.window, platform_output);

                let clipped_primitives = self.egui_ctx.tessellate(shapes, pixels_per_point);

                // --- Render Frame ---
                match self.renderer.render_frame(
                    pixels_per_point,
                    &clipped_primitives,
                    self.camera_info,
                ) {
                    Ok(_) => {
                        self.window.request_redraw();
                    }
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

                self.window.as_ref().request_redraw();
            }
            _ => {}
        }
    }

    // --- New Camera Update Function ---
    fn update_camera(&mut self, dt: f32) {
        if !self.capture_mouse
            && self.mouse_delta == (0.0, 0.0)
            && !self.is_forward_pressed
            && !self.is_backward_pressed
            && !self.is_left_pressed
            && !self.is_right_pressed
            && !self.is_up_pressed
            && !self.is_down_pressed
        {
            return; // No input, no update needed
        }

        let mut cam_pos = self.camera_info.camera_pos;
        let mut cam_target = self.camera_info.camera_target;
        let cam_up = self.camera_info.camera_up; // Usually Vec3::Y

        // --- Mouse Look (Rotation) ---
        if self.capture_mouse && self.mouse_delta != (0.0, 0.0) {
            let (delta_x, delta_y) = self.mouse_delta;
            self.mouse_delta = (0.0, 0.0); // Consume the delta

            let sensitivity = self.camera_sensitivity;
            let yaw_delta = delta_x as f32 * sensitivity;
            let pitch_delta = delta_y as f32 * sensitivity;

            let forward_dir = (cam_target - cam_pos).normalize();
            let right_dir = forward_dir.cross(cam_up).normalize();
            // Recalculate up to prevent roll if needed, though cross product handles it here
            let current_up = right_dir.cross(forward_dir).normalize();

            // --- Pitch (Up/Down) ---
            // Calculate new forward direction based on pitch rotation around right axis
            let pitch_quat = glam::Quat::from_axis_angle(right_dir, -pitch_delta); // Negative for standard mouse look
            let mut new_forward = pitch_quat * forward_dir;

            // Clamp pitch to avoid flipping over (e.g., +/- 89 degrees)
            let max_pitch_angle = 89.0f32.to_radians();
            let current_pitch = new_forward.angle_between(cam_up) - 90.0f32.to_radians();
            if current_pitch.abs() > max_pitch_angle {
                // Revert pitch if it exceeds limits
                new_forward = forward_dir; // Keep previous forward if clamp needed
            }

            // --- Yaw (Left/Right) ---
            // Rotate the (potentially pitch-adjusted) forward direction and right vector around the global up axis (Y)
            let yaw_quat = glam::Quat::from_axis_angle(Vec3::Y, -yaw_delta); // Negative for standard mouse look
            new_forward = yaw_quat * new_forward;

            // Update target based on the new forward direction
            cam_target = cam_pos + new_forward;

            // Update the camera's internal up vector based on yaw rotation as well
            // This prevents weird tilting when looking straight up/down if up wasn't Vec3::Y
            // self.camera_info.camera_up = yaw_quat * current_up; // Optional: only if up can change
        }

        // --- Keyboard Movement ---
        let forward_dir = (cam_target - cam_pos).normalize();
        // Use Vec3::Y for world-relative right/up movement, or calculate from forward/up
        let right_dir = forward_dir.cross(Vec3::Y).normalize();
        // let up_dir = right_dir.cross(forward_dir).normalize(); // Camera's local up
        let world_up_dir = Vec3::Y; // Use world up for space/shift

        let effective_speed = self.camera_speed * dt;
        let mut move_delta = Vec3::ZERO;

        if self.is_forward_pressed {
            move_delta += forward_dir;
        }
        if self.is_backward_pressed {
            move_delta -= forward_dir;
        }
        if self.is_left_pressed {
            move_delta -= right_dir;
        }
        if self.is_right_pressed {
            move_delta += right_dir;
        }
        if self.is_up_pressed {
            move_delta += world_up_dir; // Move along world Y
        }
        if self.is_down_pressed {
            move_delta -= world_up_dir; // Move along world Y
        }

        // Normalize move_delta if non-zero to ensure consistent speed diagonally
        if move_delta != Vec3::ZERO {
            let move_vec = move_delta.normalize() * effective_speed;
            cam_pos += move_vec;
            cam_target += move_vec; // Move target along with position
        }

        // --- Apply Changes ---
        self.camera_info.camera_pos = cam_pos;
        self.camera_info.camera_target = cam_target;
        // self.camera_info.camera_up remains Vec3::Y usually
    }
}

// --- Helper Functions ---

/// Finds queue family indices for graphics and presentation.
fn find_suitable_device_and_queues(
    physical_device: &PhysicalDevice,
    surface: &Surface,
    required_extensions: &[&CStr],
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

/// Game Engine
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Whether or not to create debug log (default false)
    #[arg(short, long, default_value_t = false)]
    debug_log: bool,
}

// --- Entry Point ---
fn main() -> Result<(), Box<dyn Error>> {
    color_eyre::install()?;
    let args = Args::parse();

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(true)
        .with_file(false)
        .with_line_number(false)
        .with_filter(filter::LevelFilter::DEBUG);

    let registry = tracing_subscriber::registry().with(fmt_layer);

    if args.debug_log {
        let log_file = OpenOptions::new()
            .append(true)
            .create(true)
            .open("log-debug.log")?;

        let json_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .without_time()
            .with_writer(log_file)
            .with_filter(filter::LevelFilter::DEBUG);

        registry.with(json_layer).init();
    } else {
        registry.init();
    }

    // --- Winit Setup ---
    let event_loop = EventLoop::new()?;

    // --- Event Loop ---
    info!("Starting event loop...");
    let mut app = ApplicationWrapper::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}
