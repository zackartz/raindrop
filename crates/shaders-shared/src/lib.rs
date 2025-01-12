#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::{Mat4, Vec3, Vec4};

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct Material {
    pub base_color: Vec4,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub _padding: [f32; 2],
}

#[repr(C, align(16))]
#[derive(Clone)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
    pub camera_pos: Vec3,
    pub material: Material,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PushConstants {
    pub texture_size: Vec4,
}

unsafe impl bytemuck::Pod for PushConstants {}
unsafe impl bytemuck::Zeroable for PushConstants {}
