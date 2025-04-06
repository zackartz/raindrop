mod error;

use ash::vk;
pub use error::{Result, SceneError};
use glam::Mat4;
use shared::Vertex;

use std::{collections::HashMap, path::Path, sync::Arc};

use resource_manager::{Geometry, Material, ResourceManager, SamplerDesc, SamplerHandle, Texture};

/// Represents a drawable entity in the scene, storing geometry with its transform.
#[derive(Clone)]
pub struct Mesh {
    pub name: String,
    pub material: Arc<Material>,
    pub geometry: Arc<Geometry>,
    pub transform: Mat4,
}

/// Stores all objects to be rendered by the renderer.
pub struct Scene {
    pub name: String,
    pub meshes: Vec<Mesh>,
}

fn sampler_desc_from_gltf(g_sampler: &gltf::texture::Sampler) -> SamplerDesc {
    let wrap_s = g_sampler.wrap_s();
    let wrap_t = g_sampler.wrap_t();

    SamplerDesc {
        mag_filter: g_sampler
            .mag_filter()
            .map_or(vk::Filter::LINEAR, |mf| match mf {
                gltf::texture::MagFilter::Nearest => vk::Filter::NEAREST,
                gltf::texture::MagFilter::Linear => vk::Filter::LINEAR,
            }),
        min_filter: g_sampler
            .min_filter()
            .map_or(vk::Filter::LINEAR, |mf| match mf {
                gltf::texture::MinFilter::Nearest
                | gltf::texture::MinFilter::NearestMipmapNearest
                | gltf::texture::MinFilter::NearestMipmapLinear => vk::Filter::NEAREST,
                gltf::texture::MinFilter::Linear
                | gltf::texture::MinFilter::LinearMipmapNearest
                | gltf::texture::MinFilter::LinearMipmapLinear => vk::Filter::LINEAR,
            }),
        mipmap_mode: g_sampler
            .min_filter()
            .map_or(vk::SamplerMipmapMode::LINEAR, |mf| match mf {
                gltf::texture::MinFilter::NearestMipmapNearest
                | gltf::texture::MinFilter::LinearMipmapNearest => vk::SamplerMipmapMode::NEAREST,
                gltf::texture::MinFilter::NearestMipmapLinear
                | gltf::texture::MinFilter::LinearMipmapLinear => vk::SamplerMipmapMode::LINEAR,
                _ => vk::SamplerMipmapMode::LINEAR, // Default if no mipmapping
            }),
        address_mode_u: vk_address_mode(wrap_s),
        address_mode_v: vk_address_mode(wrap_t),
        address_mode_w: vk::SamplerAddressMode::REPEAT, // glTF doesn't define wrapR
    }
}

fn vk_address_mode(g_mode: gltf::texture::WrappingMode) -> vk::SamplerAddressMode {
    match g_mode {
        gltf::texture::WrappingMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        gltf::texture::WrappingMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        gltf::texture::WrappingMode::Repeat => vk::SamplerAddressMode::REPEAT,
    }
}

impl Scene {
    /// Takes a glTF file and returns a `Scene`.
    pub fn from_gltf<T>(path: T, resource_manager: Arc<ResourceManager>) -> Result<Self>
    where
        T: AsRef<Path>,
    {
        let path_ref = path.as_ref();
        let base_path = path_ref.parent().unwrap_or_else(|| Path::new(""));
        tracing::info!("Loading glTF from: {:?}", path_ref);
        tracing::info!("Base path for resources: {:?}", base_path);

        // Import images as well
        let (doc, buffers, images) = gltf::import(path_ref)?;
        tracing::info!(
            "glTF Stats: {} scenes, {} nodes, {} meshes, {} materials, {} textures, {} images",
            doc.scenes().len(),
            doc.nodes().len(),
            doc.meshes().len(),
            doc.materials().len(),
            doc.textures().len(),
            doc.images().len()
        );

        let mut meshes = Vec::new();
        // Cache Geometry: Key = (mesh_index, primitive_index)
        let mut geometry_cache: HashMap<(usize, usize), Arc<Geometry>> = HashMap::new();
        // Cache Materials: Key = glTF material index (usize::MAX for default)
        let mut material_cache: HashMap<usize, Arc<Material>> = HashMap::new();
        // Cache default sampler handle to avoid repeated lookups
        let default_sampler_handle =
            resource_manager.get_or_create_sampler(&SamplerDesc::default())?;

        let scene_to_load = doc
            .default_scene()
            .unwrap_or_else(|| doc.scenes().next().expect("No scenes found in glTF"));

        let scene_name = scene_to_load
            .name()
            .unwrap_or("<Default Scene>")
            .to_string();
        tracing::info!(
            "Processing scene '{}' ({})",
            scene_name,
            scene_to_load.index()
        );

        // Create a context struct to pass around common data
        let mut load_ctx = LoadContext {
            doc: &doc,
            buffers: &buffers,
            images: &images,
            base_path,
            resource_manager,
            geometry_cache: &mut geometry_cache,
            material_cache: &mut material_cache,
            default_sampler_handle,
            meshes: &mut meshes,
        };

        for node in scene_to_load.nodes() {
            Self::process_node(&node, &Mat4::IDENTITY, &mut load_ctx)?;
        }

        tracing::info!("Successfully loaded {} render meshes.", meshes.len());

        Ok(Self {
            name: scene_name,
            meshes,
        })
    }

    /// Recursively processes a glTF node.
    fn process_node(
        node: &gltf::Node,
        parent_transform: &Mat4,
        ctx: &mut LoadContext, // Pass context mutably for caches
    ) -> Result<()> {
        let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
        let world_transform = *parent_transform * local_transform;
        let node_name = node.name().unwrap_or("<Unnamed Node>");

        if let Some(mesh) = node.mesh() {
            let mesh_index = mesh.index();
            let mesh_name = mesh.name().unwrap_or("<Unnamed Mesh>");
            tracing::debug!(
                "Node '{}' ({}) has Mesh '{}' ({})",
                node_name,
                node.index(),
                mesh_name,
                mesh_index
            );

            // Process mesh primitives
            for (primitive_index, primitive) in mesh.primitives().enumerate() {
                // Generate a name for the Mesh object
                let primitive_name = format!("{}_prim{}", mesh_name, primitive_index);

                Self::process_primitive(
                    &primitive,
                    mesh_index,
                    primitive_index,
                    &primitive_name, // Pass name
                    world_transform,
                    ctx, // Pass context
                )?;
            }
        } else {
            tracing::trace!("Node '{}' ({}) has no mesh.", node_name, node.index());
        }

        // Recursively process child nodes
        for child_node in node.children() {
            Self::process_node(
                &child_node,
                &world_transform, // Pass current world transform
                ctx,              // Pass context
            )?;
        }

        Ok(())
    }

    /// Processes a single glTF primitive, creating Geometry, Material, and Mesh.
    fn process_primitive(
        primitive: &gltf::Primitive,
        mesh_index: usize,
        primitive_index: usize,
        mesh_name: &str, // Name for the final Mesh object
        world_transform: Mat4,
        ctx: &mut LoadContext, // Use context
    ) -> Result<()> {
        let geometry_cache_key = (mesh_index, primitive_index);

        // --- Get or Create Geometry ---
        let geometry = if let Some(cached_geo) = ctx.geometry_cache.get(&geometry_cache_key) {
            tracing::trace!("Using cached Geometry for key {:?}", geometry_cache_key);
            cached_geo.clone()
        } else {
            tracing::trace!("Creating new Geometry for key {:?}", geometry_cache_key);
            let reader = primitive.reader(|buffer| Some(&ctx.buffers[buffer.index()]));

            let Some(pos_iter) = reader.read_positions() else {
                tracing::warn!(
                    "Primitive {:?} missing positions. Skipping.",
                    geometry_cache_key
                );
                return Ok(()); // Skip this primitive
            };
            let positions: Vec<[f32; 3]> = pos_iter.collect();
            let vertex_count = positions.len();

            if vertex_count == 0 {
                tracing::warn!(
                    "Primitive {:?} has no vertices. Skipping.",
                    geometry_cache_key
                );
                return Ok(());
            }

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| {
                    tracing::debug!(
                        "Primitive {:?} missing normals, using default.",
                        geometry_cache_key
                    );
                    vec![[0.0, 1.0, 0.0]; vertex_count]
                });

            // Read Texture Coordinates (Set 0) - needed for vertex struct regardless of material
            let tex_coords: Vec<[f32; 2]> = reader
                .read_tex_coords(0) // Read UV set 0
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| {
                    tracing::trace!(
                        "Primitive {:?} missing tex_coords (set 0), using default.",
                        geometry_cache_key
                    );
                    vec![[0.0, 0.0]; vertex_count]
                });

            if normals.len() != vertex_count || tex_coords.len() != vertex_count {
                return Err(SceneError::InconsistentData(format!(
                    "Attribute count mismatch for Primitive {:?} (Pos: {}, Norm: {}, TexCoord0: {}).",
                    geometry_cache_key, vertex_count, normals.len(), tex_coords.len()
                )));
            }

            let vertices: Vec<Vertex> = positions
                .into_iter()
                .zip(normals)
                .zip(tex_coords)
                .map(|((pos, normal), tex_coord)| Vertex {
                    pos,
                    normal,
                    tex_coord,
                })
                .collect();

            let indices: Vec<u32> = reader
                .read_indices()
                .map(|read_indices| read_indices.into_u32().collect())
                .unwrap_or_else(|| (0..vertex_count as u32).collect());

            if indices.is_empty() && vertex_count > 0 {
                tracing::warn!(
                    "Primitive {:?} has vertices but no indices. Skipping.",
                    geometry_cache_key
                );
                return Ok(());
            }

            let new_geo = Arc::new(Geometry::new(
                ctx.resource_manager.clone(),
                &vertices,
                &indices,
            )?);
            ctx.geometry_cache
                .insert(geometry_cache_key, new_geo.clone());
            new_geo
        };

        // --- Get or Create Material ---
        let g_material = primitive.material();
        // Use index usize::MAX as key for default material if index() is None
        let material_cache_key = g_material.index().unwrap_or(usize::MAX);
        let material_name = g_material.name().unwrap_or("<Default Material>");

        let material = if let Some(cached_mat) = ctx.material_cache.get(&material_cache_key) {
            tracing::trace!(
                "Using cached Material index {} ('{}')",
                material_cache_key,
                material_name
            );
            cached_mat.clone()
        } else {
            tracing::trace!(
                "Creating new Material for index {} ('{}')",
                material_cache_key,
                material_name
            );
            let pbr = g_material.pbr_metallic_roughness();

            let base_color_factor = pbr.base_color_factor();
            let metallic_factor = pbr.metallic_factor();
            let roughness_factor = pbr.roughness_factor();

            let mut loaded_base_color_texture: Option<Arc<Texture>> = None;
            let mut loaded_base_color_sampler: Option<SamplerHandle> = None;

            // --- Load Base Color Texture (if it exists) ---
            if let Some(color_info) = pbr.base_color_texture() {
                let tex_coord_set = color_info.tex_coord();
                if tex_coord_set != 0 {
                    tracing::warn!(
                        "Material '{}' requests tex_coord set {}, but only set 0 is currently loaded. Texture ignored.",
                        material_name, tex_coord_set
                    );
                    // Fall through, texture won't be loaded
                } else {
                    let g_texture = color_info.texture();
                    let g_sampler = g_texture.sampler();
                    let g_image = g_texture.source(); // This is the gltf::Image

                    tracing::debug!(
                        "Material '{}' uses Texture index {}, Sampler index {:?}, Image index {}",
                        material_name,
                        g_texture.index(),
                        g_sampler.index(),
                        g_image.index()
                    );

                    // Get or create sampler
                    let sampler_desc = sampler_desc_from_gltf(&g_sampler);
                    let sampler_handle =
                        ctx.resource_manager.get_or_create_sampler(&sampler_desc)?;
                    loaded_base_color_sampler = Some(sampler_handle);

                    // Load texture image data via ResourceManager
                    // Pass the correct gltf::Image using its index
                    let texture = ctx.resource_manager.load_texture(
                        &ctx.doc
                            .images()
                            .nth(g_image.index())
                            .expect("Image index out of bounds"), // Get gltf::Image
                        &g_image.source(), // Get gltf::image::Source
                        ctx.base_path,
                        ctx.buffers,
                        vk::ImageUsageFlags::SAMPLED, // Standard usage for textures
                    )?;
                    loaded_base_color_texture = Some(texture); // Store Arc<Texture>
                }
            }

            // Assign default sampler if none was loaded via texture info
            if loaded_base_color_sampler.is_none() {
                loaded_base_color_sampler = Some(ctx.default_sampler_handle);
                tracing::trace!("Material '{}' using default sampler.", material_name);
            }

            // Create the application Material struct
            let new_mat = Arc::new(Material {
                name: material_name.to_string(),
                base_color_texture: loaded_base_color_texture,
                base_color_sampler: loaded_base_color_sampler,
                base_color_factor,
                metallic_factor,
                roughness_factor,
                // Initialize other material properties here...
            });

            ctx.material_cache
                .insert(material_cache_key, new_mat.clone());
            new_mat
        };

        // Create the final Mesh object
        ctx.meshes.push(Mesh {
            name: mesh_name.to_string(),
            geometry,
            material, // Assign the Arc<Material>
            transform: world_transform,
        });

        Ok(())
    }
}

// Context struct to avoid passing too many arguments
struct LoadContext<'a> {
    doc: &'a gltf::Document,
    buffers: &'a [gltf::buffer::Data],
    images: &'a [gltf::image::Data], // Keep image data accessible if needed by RM
    base_path: &'a Path,
    resource_manager: Arc<ResourceManager>,
    geometry_cache: &'a mut HashMap<(usize, usize), Arc<Geometry>>,
    material_cache: &'a mut HashMap<usize, Arc<Material>>,
    default_sampler_handle: SamplerHandle, // Store the default sampler
    meshes: &'a mut Vec<Mesh>,             // Store results directly
}
