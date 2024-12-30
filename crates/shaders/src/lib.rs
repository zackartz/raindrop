#![cfg_attr(target_arch = "spirv", no_std)]

use shaders_shared::UniformBufferObject;
use spirv_std::{
    glam::{Mat3, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles},
    image::Image2d,
    spirv, Sampler,
};

#[spirv(vertex)]
pub fn main_vs(
    // Vertex inputs
    in_pos: Vec3,
    in_normal: Vec3,
    in_tex_coord: Vec2,

    // Uniform buffer
    #[spirv(uniform, descriptor_set = 0, binding = 0)] ubo: &UniformBufferObject,

    // Vertex outputs
    out_world_position: &mut Vec3,
    out_world_normal: &mut Vec3,
    out_tex_coord: &mut Vec2,
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
    *out_tex_coord = in_tex_coord;
}

pub fn get_uv_u(pix: UVec2, tex_size: Vec4) -> spirv_std::glam::Vec2 {
    (pix.as_vec2() + Vec2::splat(0.5)) * tex_size.zw()
}

#[spirv(fragment)]
pub fn main_fs(
    frag_world_position: Vec3,
    frag_world_normal: Vec3,
    frag_tex_coord: Vec2,
    #[spirv(descriptor_set = 0, binding = 1)] texture: &Image2d,
    #[spirv(descriptor_set = 0, binding = 2)] sampler: &Sampler,
    out_color: &mut Vec4,
) {
    // Convert fragment coordinate to UV coordinate
    // Sample the texture
    let base_color = texture.sample(*sampler, frag_tex_coord);
    // let light_pos = Vec3::new(2.0, 2.0, -2.0);
    //
    // // Calculate light direction
    // let l = (light_pos - frag_world_position).normalize();
    // let n = frag_world_normal.normalize();
    //
    // // Calculate lambertian lighting
    // let lambertian = f32::max(n.dot(l), 0.0);
    //
    // // Ambient lighting
    // let ambient = Vec3::splat(0.2);
    //
    // // Combine texture color with model color
    // let color_from_texture = Vec3::new(base_color.x, base_color.y, base_color.z);
    // let combined_color = color_from_texture;
    //
    // // Final color calculation with gamma correction
    // let color = (combined_color * lambertian + ambient).powf(2.2);
    *out_color = Vec4::new(base_color.x, base_color.y, base_color.z, base_color.w);
}
