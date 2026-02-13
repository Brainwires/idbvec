#!/bin/bash
set -e

echo "Building idbvec for WebAssembly..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build for different targets
echo ""
echo "Building for bundler (webpack, rollup, vite, etc.)..."
wasm-pack build --target bundler --out-dir pkg/bundler

echo ""
echo "Building for Node.js..."
wasm-pack build --target nodejs --out-dir pkg/nodejs

echo ""
echo "Building for web (ES modules)..."
wasm-pack build --target web --out-dir pkg/web

echo ""
echo "✓ WASM build complete!"
echo ""
echo "Output directories:"
echo "  - pkg/bundler  (for webpack/rollup/vite)"
echo "  - pkg/nodejs   (for Node.js)"
echo "  - pkg/web      (for ES modules)"
echo ""
