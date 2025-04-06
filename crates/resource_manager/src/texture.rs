use std::sync::Arc;

use ash::vk;

use crate::{ImageHandle, SamplerHandle};

#[derive(Debug, Clone)]
pub struct Texture {
    pub handle: ImageHandle,

    pub format: vk::Format,
    pub extent: vk::Extent3D,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SamplerDesc {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub base_color_texture: Option<Arc<Texture>>,
    pub base_color_sampler: Option<SamplerHandle>,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    // TODO: Add other PBR properties:
    // pub metallic_roughness_texture: Option<Arc<Texture>>,
    // pub metallic_roughness_sampler: Option<SamplerHandle>,
    // pub normal_texture: Option<Arc<Texture>>,
    // pub normal_sampler: Option<SamplerHandle>,
    // pub occlusion_texture: Option<Arc<Texture>>,
    // pub occlusion_sampler: Option<SamplerHandle>,
    // pub emissive_texture: Option<Arc<Texture>>,
    // pub emissive_sampler: Option<SamplerHandle>,
    // pub emissive_factor: [f32; 3],
    // pub alpha_mode: gltf::material::AlphaMode,
    // pub alpha_cutoff: f32,
    // pub double_sided: bool,
}
