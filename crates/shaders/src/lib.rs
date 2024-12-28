#![cfg_attr(target_arch = "spirv", no_std)]

use shaders_shared::UniformBufferObject;
use spirv_std::spirv;

use glam::{Mat3, Vec3, Vec4, Vec4Swizzles};

#[spirv(vertex)]
pub fn main_vs(
    // Vertex inputs
    in_pos: Vec3,
    in_normal: Vec3,

    // Uniform buffer
    #[spirv(uniform, descriptor_set = 0, binding = 0)] ubo: &UniformBufferObject,

    // Vertex outputs
    out_world_position: &mut Vec3,
    out_world_normal: &mut Vec3,
    #[spirv(position)] gl_position: &mut Vec4,
) {
    // Transform position to world space
    let pos = ubo.model * Vec4::from((in_pos, 1.0));
    *out_world_position = (pos / pos.w).xyz();

    // Transform normal to world space
    let normal_matrix = Mat3::from_mat4(ubo.model).inverse().transpose();
    *out_world_normal = normal_matrix * in_normal;

    // Calculate clip space position
    *gl_position = ubo.proj * ubo.view * pos;
}

#[spirv(fragment)]
pub fn main_fs(frag_world_position: Vec3, frag_world_normal: Vec3, out_color: &mut Vec4) {
    let base_color = Vec3::new(1.0, 0.5, 0.5);
    let light_pos = Vec3::new(2.0, 2.0, -2.0);

    // Calculate light direction
    let l = (light_pos - frag_world_position).normalize();
    let n = frag_world_normal.normalize();

    // Calculate lambertian lighting
    let lambertian = f32::max(n.dot(l), 0.0);

    // Ambient lighting
    let ambient = Vec3::splat(0.1);

    // Final color calculation with gamma correction
    let color = (base_color * lambertian + ambient).powf(2.2);
    *out_color = Vec4::from((color, 1.0));
}
