use std::{
    collections::HashSet,
    ffi::CString,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk::{self, KhrAccelerationStructureFn, KhrDeferredHostOperationsFn, KhrRayTracingPipelineFn},
    Device, Entry, Instance,
};
use egui_ash::{
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
    winit, App, AppCreator, AshRenderState, CreationContext, HandleRedraw, RunOption, Theme,
};
use glam::Vec3;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use renderer::Renderer;

mod renderer;

struct Game {
    entry: Entry,
    instance: Instance,
    device: Device,
    debug_utils_loader: DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    surface_loader: Surface,
    swapchain_loader: Swapchain,
    surface: vk::SurfaceKHR,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,

    pub renderer: Renderer,

    theme: Theme,
    rotate_y: f32,

    camera_position: Vec3,
    camera_yaw: f32,
    camera_pitch: f32,
    right_mouse_pressed: bool,
    last_mouse_pos: Option<(f32, f32)>,

    last_fps_update: std::time::Instant,
    frame_count_since_last_update: i32,
    current_fps: f32,
}

impl App for Game {
    fn ui(&mut self, ctx: &egui::Context) {
        let now = std::time::Instant::now();
        self.frame_count_since_last_update += 1;

        if now.duration_since(self.last_fps_update).as_secs_f32() >= 0.1 {
            self.current_fps = self.frame_count_since_last_update as f32
                / now.duration_since(self.last_fps_update).as_secs_f32();
            self.frame_count_since_last_update = 0;
            self.last_fps_update = now;
        }

        egui::SidePanel::left("my_side_panel").show(ctx, |ui| {
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Theme");
                let id = ui.make_persistent_id("theme_combo_box_side");
                egui::ComboBox::from_id_source(id)
                    .selected_text(format!("{:?}", self.theme))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.theme, Theme::Dark, "Dark");
                        ui.selectable_value(&mut self.theme, Theme::Light, "Light");
                    });
            });
            ui.separator();
            ui.label("Rotate");
            ui.add(egui::widgets::Slider::new(
                &mut self.rotate_y,
                -180.0..=180.0,
            ));
            ui.separator();
            ui.label("Camera Position");
            ui.horizontal(|ui| {
                ui.label("X:");
                ui.add(egui::DragValue::new(&mut self.camera_position.x).speed(0.1));
            });
            ui.horizontal(|ui| {
                ui.label("Y:");
                ui.add(egui::DragValue::new(&mut self.camera_position.y).speed(0.1));
            });
            ui.horizontal(|ui| {
                ui.label("Z:");
                ui.add(egui::DragValue::new(&mut self.camera_position.z).speed(0.1));
            });
            ui.label(format!("FPS: {:.1}", self.current_fps));
        });

        if !ctx.wants_keyboard_input() {
            let movement_speed = 0.1;

            let forward = Vec3::new(self.camera_yaw.sin(), 0.0, self.camera_yaw.cos()).normalize();

            let right = Vec3::new(
                (self.camera_yaw + 90.0_f32.to_radians()).sin(),
                0.0,
                (self.camera_yaw + 90.0_f32.to_radians()).cos(),
            )
            .normalize();

            ctx.input(|i| {
                if i.key_down(egui::Key::W) {
                    self.camera_position += forward * movement_speed;
                }
                if i.key_down(egui::Key::S) {
                    self.camera_position -= forward * movement_speed;
                }
                if i.key_down(egui::Key::A) {
                    self.camera_position -= right * movement_speed;
                }
                if i.key_down(egui::Key::D) {
                    self.camera_position += right * movement_speed;
                }
            });
        }

        // Handle mouse input for camera rotation
        let is_right_mouse_down = ctx.input(|i| i.pointer.secondary_down());
        let hover_pos = ctx.input(|i| i.pointer.hover_pos());

        // Set cursor visibility based on right mouse button
        if is_right_mouse_down != self.right_mouse_pressed {
            if is_right_mouse_down {
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(false));
            } else {
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
            }
        }

        self.right_mouse_pressed = is_right_mouse_down;

        if self.right_mouse_pressed {
            if let Some(pos) = hover_pos {
                if let Some((last_x, last_y)) = self.last_mouse_pos {
                    let delta_x = pos.x - last_x;
                    let delta_y = pos.y - last_y;

                    // Update camera rotation
                    let rotation_speed = 0.002;
                    self.camera_yaw -= delta_x * rotation_speed;
                    self.camera_pitch = (self.camera_pitch + delta_y * rotation_speed)
                        .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
                }
                self.last_mouse_pos = Some((pos.x, pos.y));
            }
        } else {
            self.last_mouse_pos = None;
        }

        match self.theme {
            Theme::Dark => ctx.set_visuals(egui::style::Visuals::dark()),
            Theme::Light => ctx.set_visuals(egui::style::Visuals::light()),
        }
    }

    fn request_redraw(&mut self, _viewport_id: egui::ViewportId) -> HandleRedraw {
        HandleRedraw::Handle(Box::new({
            let renderer = self.renderer.clone();
            let rotate_y = self.rotate_y;
            let camera_position = self.camera_position;
            let camera_yaw = self.camera_yaw;
            let camera_pitch = self.camera_pitch;
            move |size, egui_cmd| {
                let mut renderer = renderer.inner.lock().unwrap();
                renderer.update_camera(camera_position, camera_yaw, camera_pitch);
                renderer.render(size.width, size.height, egui_cmd, rotate_y)
            }
        }))
    }
}

impl Drop for Game {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.renderer.destroy();
            self.device.destroy_command_pool(self.command_pool, None);
            self.surface_loader.destroy_surface(self.surface, None);
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

struct MyAppCreator;
impl MyAppCreator {
    const ENABLE_VALIDATION_LAYERS: bool = true;
    const VALIDATION: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];

    unsafe extern "system" fn vulkan_debug_utils_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_types: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        let severity = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[VERBOSE]",
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[WARNING]",
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[ERROR]",
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[INFO]",
            _ => panic!("[UNKNOWN]"),
        };
        let types = match message_types {
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[GENERAL]",
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[PERFORMANCE]",
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[VALIDATION]",
            _ => panic!("[UNKNOWN]"),
        };
        let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
        println!("[DEBUG]{}{}{:?}", severity, types, message);

        vk::FALSE
    }

    fn create_entry() -> Entry {
        unsafe { Entry::load().unwrap() }
    }

    fn create_instance(
        required_instance_extensions: &[CString],
        entry: &Entry,
    ) -> (Instance, DebugUtils, vk::DebugUtilsMessengerEXT) {
        let mut debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(Self::vulkan_debug_utils_callback))
            .build();

        let app_name = std::ffi::CString::new("egui-winit-ash example simple").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_api_version(1, 0, 0, 0))
            .api_version(vk::API_VERSION_1_2);
        let mut extension_names = vec![DebugUtils::name().as_ptr()];
        for ext in required_instance_extensions {
            let name = ext.as_ptr();
            extension_names.push(name);
        }
        let raw_layer_names = Self::VALIDATION
            .iter()
            .map(|l| std::ffi::CString::new(*l).unwrap())
            .collect::<Vec<_>>();
        let layer_names = raw_layer_names
            .iter()
            .map(|l| l.as_ptr())
            .collect::<Vec<*const i8>>();
        let instance_create_info = vk::InstanceCreateInfo::builder()
            .push_next(&mut debug_utils_messenger_create_info)
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);
        let instance_create_info = if Self::ENABLE_VALIDATION_LAYERS {
            instance_create_info.enabled_layer_names(&layer_names)
        } else {
            instance_create_info
        };
        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failed to create instance")
        };

        // setup debug utils
        let debug_utils_loader = DebugUtils::new(entry, &instance);
        let debug_messenger = if Self::ENABLE_VALIDATION_LAYERS {
            unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)
                    .expect("Failed to create debug utils messenger")
            }
        } else {
            vk::DebugUtilsMessengerEXT::null()
        };

        (instance, debug_utils_loader, debug_messenger)
    }

    fn create_surface_loader(entry: &Entry, instance: &Instance) -> Surface {
        Surface::new(entry, instance)
    }

    fn create_swapchain_loader(instance: &Instance, device: &Device) -> Swapchain {
        Swapchain::new(instance, device)
    }

    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        window: &winit::window::Window,
    ) -> vk::SurfaceKHR {
        unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .expect("Failed to create surface")
        }
    }

    fn create_physical_device(
        instance: &Instance,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        required_device_extensions: &[CString],
    ) -> (vk::PhysicalDevice, vk::PhysicalDeviceMemoryProperties, u32) {
        let mut queue_family_index: Option<usize> = None;

        let (physical_device, physical_device_memory_properties) = {
            let physical_devices = unsafe {
                instance
                    .enumerate_physical_devices()
                    .expect("Failed to enumerate physical devices")
            };
            let physical_device = physical_devices.into_iter().find(|physical_device| {
                let queue_families = unsafe {
                    instance.get_physical_device_queue_family_properties(*physical_device)
                };
                for (i, queue_family) in queue_families.iter().enumerate() {
                    let mut graphics_queue = false;
                    let mut present_queue = false;
                    if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        graphics_queue = true;
                    }
                    let present_support = unsafe {
                        surface_loader
                            .get_physical_device_surface_support(
                                *physical_device,
                                i as u32,
                                surface,
                            )
                            .unwrap()
                    };
                    if present_support {
                        present_queue = true;
                    }
                    if graphics_queue && present_queue {
                        queue_family_index = Some(i);
                        break;
                    }
                }
                let is_queue_family_supported = queue_family_index.is_some();

                // check device extensions
                let device_extensions = unsafe {
                    instance
                        .enumerate_device_extension_properties(*physical_device)
                        .unwrap()
                };
                let mut device_extensions_name = vec![];
                for device_extension in device_extensions {
                    let name = unsafe {
                        std::ffi::CStr::from_ptr(device_extension.extension_name.as_ptr())
                            .to_owned()
                    };
                    device_extensions_name.push(name);
                }
                let mut required_extensions = HashSet::new();
                for extension in required_device_extensions.iter() {
                    required_extensions.insert(extension.to_owned());
                }
                for extension_name in device_extensions_name {
                    required_extensions.remove(&extension_name);
                }
                let is_device_extension_supported = required_extensions.is_empty();

                // check swapchain support
                let surface_formats = unsafe {
                    surface_loader
                        .get_physical_device_surface_formats(*physical_device, surface)
                        .unwrap()
                };
                let surface_present_modes = unsafe {
                    surface_loader
                        .get_physical_device_surface_present_modes(*physical_device, surface)
                        .unwrap()
                };
                let is_swapchain_supported =
                    !surface_formats.is_empty() && !surface_present_modes.is_empty();

                is_queue_family_supported && is_swapchain_supported && is_device_extension_supported
            });
            let physical_device = physical_device.expect("Failed to get physical device");
            let physical_device_memory_properties =
                unsafe { instance.get_physical_device_memory_properties(physical_device) };

            (physical_device, physical_device_memory_properties)
        };

        (
            physical_device,
            physical_device_memory_properties,
            queue_family_index.unwrap() as u32,
        )
    }

    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        required_device_extensions: &[CString],
    ) -> (Device, vk::Queue) {
        let queue_priorities = [1.0_f32];
        let mut queue_create_infos = vec![];
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)
            .build();
        queue_create_infos.push(queue_create_info);

        let physical_device_features = vk::PhysicalDeviceFeatures::builder().build();

        let enable_extension_names = required_device_extensions
            .iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<_>>();

        // device create info
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(&enable_extension_names);

        // create device
        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create device")
        };

        // get device queue
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        (device, queue)
    }

    fn create_command_pool(device: &Device, queue_family_index: u32) -> vk::CommandPool {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);
        unsafe {
            device
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create command pool")
        }
    }
}

impl AppCreator<Arc<Mutex<Allocator>>> for MyAppCreator {
    type App = Game;

    fn create(&self, cc: CreationContext) -> (Self::App, AshRenderState<Arc<Mutex<Allocator>>>) {
        // create vk objects
        let entry = Self::create_entry();
        let (instance, debug_utils_loader, debug_messenger) =
            Self::create_instance(&cc.required_instance_extensions, &entry);
        let surface_loader = Self::create_surface_loader(&entry, &instance);
        let surface = Self::create_surface(&entry, &instance, cc.main_window);
        let mut req_ext = vec![
            KhrDeferredHostOperationsFn::name().to_owned(),
            KhrRayTracingPipelineFn::name().to_owned(),
            KhrAccelerationStructureFn::name().to_owned(),
        ];
        for ext in &cc.required_device_extensions {
            req_ext.push(ext.to_owned());
        }
        let (physical_device, _physical_device_memory_properties, queue_family_index) =
            Self::create_physical_device(&instance, &surface_loader, surface, &req_ext);
        let (device, queue) = Self::create_device(
            &instance,
            physical_device,
            queue_family_index,
            &cc.required_device_extensions,
        );
        let swapchain_loader = Self::create_swapchain_loader(&instance, &device);
        let command_pool = Self::create_command_pool(&device, queue_family_index);

        // create allocator
        let allocator = {
            Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .expect("Failed to create allocator")
        };
        let allocator = Arc::new(Mutex::new(allocator));

        // setup context
        cc.context.set_visuals(egui::style::Visuals::dark());

        let app = Game {
            entry,
            instance,
            device: device.clone(),
            debug_utils_loader,
            debug_messenger,
            physical_device,
            surface_loader: surface_loader.clone(),
            swapchain_loader: swapchain_loader.clone(),
            surface,
            queue,
            command_pool,
            allocator: ManuallyDrop::new(allocator.clone()),

            renderer: Renderer::new(
                physical_device,
                device,
                surface_loader,
                swapchain_loader,
                allocator.clone(),
                surface,
                queue_family_index,
                queue,
                command_pool,
                1000,
                800,
            ),

            theme: if cc.context.style().visuals.dark_mode {
                Theme::Dark
            } else {
                Theme::Light
            },
            rotate_y: 0.0,

            camera_position: Vec3::new(0.0, 0.0, -5.0),
            camera_pitch: 0.,
            camera_yaw: 0.,
            last_mouse_pos: None,
            right_mouse_pressed: false,
            last_fps_update: std::time::Instant::now(),
            frame_count_since_last_update: 0,
            current_fps: 0.0,
        };
        let ash_render_state = AshRenderState {
            entry: app.entry.clone(),
            instance: app.instance.clone(),
            physical_device: app.physical_device,
            device: app.device.clone(),
            surface_loader: app.surface_loader.clone(),
            swapchain_loader: app.swapchain_loader.clone(),
            queue: app.queue,
            queue_family_index,
            command_pool: app.command_pool,
            allocator: allocator.clone(),
        };

        (app, ash_render_state)
    }
}

fn main() -> std::process::ExitCode {
    egui_ash::run(
        "vulkan",
        MyAppCreator,
        RunOption {
            viewport_builder: Some(
                egui::ViewportBuilder::default()
                    .with_title("vulkan")
                    .with_inner_size(egui::vec2(1000.0, 800.0)),
            ),
            follow_system_theme: false,
            default_theme: Theme::Dark,
            ..Default::default()
        },
    )
}
