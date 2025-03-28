{
  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/default";

    treefmt-nix.url = "github:numtide/treefmt-nix";

    rust-overlay.url = "github:oxalica/rust-overlay";
    crane.url = "github:ipetkov/crane";
  };

  # Add settings for your binary cache.
  nixConfig = {
  };

  outputs = inputs @ {
    nixpkgs,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import inputs.systems;

      imports = [
        inputs.treefmt-nix.flakeModule
      ];

      perSystem = {
        config,
        system,
        pkgs,
        lib,
        craneLib,
        commonArgs,
        ...
      }: {
        _module.args = {
          pkgs = import nixpkgs {
            inherit system;
            overlays = [inputs.rust-overlay.overlays.default];
          };
          craneLib = (inputs.crane.mkLib pkgs).overrideToolchain (
            pkgs: pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml
          );
          commonArgs = {
            # Depending on your code base, you may have to customize the
            # source filtering to include non-standard files during the build.
            # See
            # https://crane.dev/source-filtering.html?highlight=source#source-filtering
            pname = "vk-rs";
            version = "0.1.0";
            src = craneLib.cleanCargoSource (craneLib.path ./.);

            nativeBuildInputs = with pkgs; [
              pkg-config
              vulkan-tools
              vulkan-headers
              vulkan-loader
              vulkan-validation-layers
              cmake
            ];

            buildInputs = with pkgs; [
              libxkbcommon
              libGL
              python3
              # WINIT_UNIX_BACKEND=wayland
              wayland
              spirv-tools
              spirv-cross
              vulkan-loader
            ];
          };
        };

        # Build the executable package.
        packages.default = craneLib.buildPackage (
          commonArgs
          // {
            cargoArtifacts = craneLib.buildDepsOnly commonArgs;
          }
        );

        devShells.default = craneLib.devShell {
          packages =
            (commonArgs.nativeBuildInputs or [])
            ++ (commonArgs.buildInputs or [])
            ++ [pkgs.rust-analyzer-unwrapped pkgs.cargo-workspaces];

          hardeningDisable = ["fortify"];

          RUST_SRC_PATH = "${
            (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml)
          }/lib/rustlib/src/rust/library";
          LD_LIBRARY_PATH = "${lib.makeLibraryPath commonArgs.buildInputs}:${pkgs.stdenv.cc.cc.lib}/lib";
        };

        treefmt = {
          projectRootFile = "Cargo.toml";
          programs = {
            actionlint.enable = true;
            nixfmt.enable = true;
            rustfmt.enable = true;
          };
        };
      };
    };
}
