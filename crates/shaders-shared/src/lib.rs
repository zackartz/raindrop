#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Clone)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
    pub model_color: Vec3,
}
