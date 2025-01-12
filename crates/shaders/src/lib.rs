#![cfg_attr(target_arch = "spirv", no_std)]

use shaders_shared::UniformBufferObject;
use spirv_std::num_traits::Float;
use spirv_std::{
    glam::{Mat3, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles},
    image::Image2d,
    spirv, Sampler,
};

pub const PI: f32 = core::f32::consts::PI;

fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    f0 + (Vec3::ONE - f0) * (1.0 - cos_theta).powf(5.0)
}

fn distribution_ggx(n: Vec3, h: Vec3, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = n.dot(h).max(0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;

    let nom = a2;
    let denom = (n_dot_h2 * (a2 - 1.0) + 1.0);
    let denom = PI * denom * denom;

    nom / denom.max(0.001)
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;

    let nom = n_dot_v;
    let denom = n_dot_v * (1.0 - k) + k;

    nom / denom
}

fn geometry_smith(n: Vec3, v: Vec3, l: Vec3, roughness: f32) -> f32 {
    let n_dot_v = n.dot(v).max(0.0);
    let n_dot_l = n.dot(l).max(0.0);
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);

    ggx1 * ggx2
}

#[spirv(vertex)]
pub fn main_vs(
    #[spirv(position)] in_pos: Vec3,
    in_normal: Vec3,
    in_tex_coord: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] ubo: &UniformBufferObject,
    out_world_pos: &mut Vec3,
    out_world_normal: &mut Vec3,
    out_tex_coord: &mut Vec2,
    #[spirv(position)] out_pos: &mut Vec4,
) {
    let pos = ubo.model * Vec4::from((in_pos, 1.0));
    *out_world_pos = pos.truncate();

    // Transform normal to world space
    let normal_matrix = Mat3::from_mat4(ubo.model).inverse().transpose();
    *out_world_normal = (normal_matrix * in_normal).normalize();

    *out_pos = ubo.proj * ubo.view * pos;
    *out_tex_coord = in_tex_coord;
}
