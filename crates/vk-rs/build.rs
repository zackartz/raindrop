use shaderc::{Compiler, ShaderKind};
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tell Cargo to rerun if shaders directory changes
    println!("cargo:rerun-if-changed=../../shaders");

    let shader_dir = Path::new("../../shaders");
    let cache_dir = Path::new("../../shader-cache");

    // Create shader cache directory if it doesn't exist
    fs::create_dir_all(cache_dir)?;

    let compiler = Compiler::new().expect("Failed to create shader compiler");

    // Compile all .vert and .frag files
    for entry in fs::read_dir(shader_dir)? {
        let entry = entry?;
        let path = entry.path();

        if let Some(extension) = path.extension() {
            let kind = match extension.to_str() {
                Some("vert") => ShaderKind::Vertex,
                Some("frag") => ShaderKind::Fragment,
                _ => continue,
            };

            let source = fs::read_to_string(&path)?;
            let file_name = path.file_name().unwrap().to_str().unwrap();

            // Create output path
            let spirv_path = cache_dir.join(format!("{}.spv", file_name));

            // Check if we need to recompile
            if should_compile(&path, &spirv_path) {
                println!("Compiling shader: {}", file_name);

                let compiled =
                    compiler.compile_into_spirv(&source, kind, file_name, "main", None)?;

                let mut file = File::create(&spirv_path)?;
                file.write_all(compiled.as_binary_u8())?;
            }
        }
    }

    Ok(())
}

fn should_compile(source_path: &Path, output_path: &PathBuf) -> bool {
    // If output doesn't exist, we need to compile
    if !output_path.exists() {
        return true;
    }

    // Get modification times
    let source_modified = fs::metadata(source_path)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let output_modified = fs::metadata(output_path)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    // Compile if source is newer than output
    source_modified > output_modified
}
