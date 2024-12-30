use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use ash::Device;
use gpu_allocator::vulkan::Allocator;

use super::renderer::Texture;

pub struct TextureCache {
    cache: HashMap<String, Arc<Texture>>,
}

impl TextureCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn get_or_load_texture(
        &mut self,
        key: String,
        load_fn: impl FnOnce() -> Option<Texture>,
    ) -> Option<Arc<Texture>> {
        if let Some(texture) = self.cache.get(&key) {
            Some(Arc::clone(texture))
        } else {
            load_fn().map(|texture| {
                let texture = Arc::new(texture);
                self.cache.insert(key, Arc::clone(&texture));
                texture
            })
        }
    }

    pub fn cleanup(&mut self, device: &Device, allocator: &mut Allocator) {
        for (_, texture) in self.cache.drain() {
            if let Ok(texture) = Arc::try_unwrap(texture) {
                let mut texture = texture;
                texture.destroy(device, allocator);
            }
        }
    }
}
