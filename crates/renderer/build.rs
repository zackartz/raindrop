use anyhow::{Context, Result};
use shaderc::{CompileOptions, Compiler, ShaderKind};
use std::{
    env,
    fs::{self, File},
    io::Write,
    path::PathBuf,
};
use walkdir::WalkDir;

// Configuration
const SHADER_SOURCE_DIR: &str = "../../shaders"; // Directory containing GLSL shaders
                                                 // Output directory will be determined by Cargo (OUT_DIR)

fn main() -> Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?).join("shaders"); // Put shaders in a subdirectory for clarity
    fs::create_dir_all(&out_dir).context("Failed to create shader output directory")?;

    let compiler = Compiler::new().context("Failed to create shader compiler")?;
    let mut options = CompileOptions::new().context("Failed to create compile options")?;

    // --- Optional: Add compile options ---
    // Example: Optimize for performance in release builds
    if env::var("PROFILE")? == "release" {
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        eprintln!("Build.rs: Compiling shaders with Performance optimization.");
    } else {
        options.set_optimization_level(shaderc::OptimizationLevel::Zero); // Faster compile for debug
        options.set_generate_debug_info(); // Include debug info for debug builds
        eprintln!("Build.rs: Compiling shaders with Zero optimization and Debug info.");
    }
    // Add other options like defines if needed:
    // options.add_macro_definition("MY_DEFINE", Some("1"));
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    ); // Specify Vulkan version if needed

    eprintln!(
        "Build.rs: Compiling shaders from '{}' to '{}'",
        SHADER_SOURCE_DIR,
        out_dir.display()
    );

    // --- Find and Compile Shaders ---
    for entry in WalkDir::new(SHADER_SOURCE_DIR)
        .into_iter()
        .filter_map(|e| e.ok()) // Ignore directory reading errors
        .filter(|e| e.file_type().is_file())
    // Only process files
    {
        let in_path = entry.path();

        // Determine shader kind from extension
        let extension = match in_path.extension().and_then(|s| s.to_str()) {
            Some(ext) => ext,
            None => {
                eprintln!(
                    "cargo:warning=Skipping file with no extension: {}",
                    in_path.display()
                );
                continue; // Skip files without extensions
            }
        };
        let shader_kind = match extension {
            "vert" => ShaderKind::Vertex,
            "frag" => ShaderKind::Fragment,
            "comp" => ShaderKind::Compute,
            "geom" => ShaderKind::Geometry,
            "tesc" => ShaderKind::TessControl,
            "tese" => ShaderKind::TessEvaluation,
            // Add other shader kinds if needed (ray tracing, mesh, etc.)
            _ => {
                eprintln!(
                    "cargo:warning=Skipping file with unknown shader extension ({}): {}",
                    extension,
                    in_path.display()
                );
                continue; // Skip unknown shader types
            }
        };

        let source_text = fs::read_to_string(in_path)
            .with_context(|| format!("Failed to read shader source: {}", in_path.display()))?;
        let input_file_name = in_path.to_string_lossy(); // For error messages

        // Compile the shader
        let compiled_spirv = compiler
            .compile_into_spirv(
                &source_text,
                shader_kind,
                &input_file_name, // Source file name for errors
                "main",           // Entry point function name
                Some(&options),   // Pass compile options
            )
            .with_context(|| format!("Failed to compile shader: {}", input_file_name))?;

        let spirv_bytes = compiled_spirv.as_binary_u8();
        let byte_count = spirv_bytes.len();
        eprintln!(
            "Build.rs: SPIR-V for {} has {} bytes.",
            input_file_name, byte_count
        );

        // Check if it's a multiple of 4 right here
        if byte_count % 4 != 0 {
            eprintln!(
                "cargo:warning=Byte count for {} ({}) is NOT a multiple of 4!",
                input_file_name, byte_count
            );
            // Optionally bail out here:
            // bail!("Generated SPIR-V for {} has invalid byte count {}", input_file_name, byte_count);
        }

        // Check for warnings
        if compiled_spirv.get_num_warnings() > 0 {
            eprintln!(
                "cargo:warning=Shader compilation warnings for {}:\n{}",
                input_file_name,
                compiled_spirv.get_warning_messages()
            );
        }

        // Determine output path
        let out_filename = format!(
            "{}.spv",
            in_path
                .file_stem() // Get filename without extension
                .unwrap_or_default() // Handle potential weird filenames
                .to_string_lossy()
        );
        let out_path = out_dir.join(out_filename);

        // Determine output path...
        // ...
        // Write the compiled SPIR-V binary
        let mut outfile = File::create(&out_path)
            .with_context(|| format!("Failed to create output file: {}", out_path.display()))?;
        outfile
            .write_all(spirv_bytes) // Use the stored bytes
            .with_context(|| format!("Failed to write SPIR-V to: {}", out_path.display()))?;

        eprintln!(
            "Build.rs: Compiled {} -> {}",
            in_path.display(),
            out_path.display()
        );
    }

    eprintln!("Build.rs: Shader compilation finished.");
    Ok(())
}
