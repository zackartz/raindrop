use ash::vk;
use glam::{Mat4, Vec3};

use core::f32;
use std::mem::size_of;

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub fn get_binding_decription() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(memoffset::offset_of!(Vertex, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(memoffset::offset_of!(Vertex, color) as u32),
        ]
    }
}

#[repr(C)]
#[derive(Clone, Debug, Copy, PartialEq)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct CameraInfo {
    pub camera_pos: Vec3,
    pub camera_target: Vec3,
    pub camera_fov: f32,
}

impl Default for CameraInfo {
    fn default() -> Self {
        Self {
            camera_pos: Vec3::new(10.0, 10.0, 10.0),
            camera_target: Vec3::new(0.0, 0.0, 0.0),
            camera_fov: 45.0,
        }
    }
}
