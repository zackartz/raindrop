#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::{Mat4, Vec3, Vec4};

#[repr(C)]
#[derive(Clone)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
    pub model_color: Vec3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PushConstants {
    pub texture_size: Vec4,
}

unsafe impl bytemuck::Pod for PushConstants {}
unsafe impl bytemuck::Zeroable for PushConstants {}
