use thiserror::Error;

/// Any errors that can be returned from this crate.
#[derive(Error, Debug)]
pub enum SceneError {
    #[error("Error from ResourceManager: {0}")]
    ResourceManagerError(#[from] resource_manager::ResourceManagerError),

    #[error("Error from GLTF: {0}")]
    GltfError(#[from] gltf::Error),

    #[error("InconsistentData: {0}")]
    InconsistentData(String),
}

pub type Result<T> = std::result::Result<T, SceneError>;
