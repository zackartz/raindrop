use std::{
    fs::{self, File},
    io::{Read, Write},
};

use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tell Cargo to rerun this script if the shaders crate or its contents change
    println!("cargo:rerun-if-changed=../shaders/src");
    println!("cargo:rerun-if-changed=../shaders/Cargo.toml");

    SpirvBuilder::new("../shaders/", "spirv-unknown-vulkan1.2")
        .print_metadata(MetadataPrintout::None)
        .multimodule(true)
        .build()?
        .module
        .unwrap_multi()
        .iter()
        .for_each(|(name, path)| {
            let mut data = vec![];
            File::open(path).unwrap().read_to_end(&mut data).unwrap();

            fs::create_dir_all("./shaders/").unwrap();

            File::create(format!("./shaders/{name}.spv"))
                .unwrap()
                .write_all(&data)
                .unwrap();
        });

    Ok(())
}
