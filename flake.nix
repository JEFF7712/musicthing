{
  description = "Neural Vibe — music discovery via brain response similarity";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python
            python312
            uv

            # System deps for TRIBE v2
            ffmpeg
            sox

            # Build tools (for native extensions)
            gcc
            stdenv.cc.cc.lib

            # CUDA
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH"
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"

            # Create venv if it doesn't exist
            if [ ! -d .venv ]; then
              uv venv --python python3.12 .venv
            fi
            source .venv/bin/activate
          '';
        };
      });
}
