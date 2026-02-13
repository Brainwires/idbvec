#!/bin/bash
set -e

echo "Building @brainwires/idbvec for npm..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build WASM (bundler target only — that's what npm consumers need)
echo "Building WASM (bundler target)..."
wasm-pack build --target bundler --out-dir pkg/bundler

# Remove wasm-pack's .gitignore (contains "*") which causes npm pack to skip the directory
rm -f pkg/bundler/.gitignore

# Compile TypeScript wrapper
echo "Compiling TypeScript wrapper..."
npx tsc

# Copy compiled files from dist/ to root for package entry point
cp dist/wrapper.js wrapper.js
cp dist/wrapper.d.ts wrapper.d.ts

echo ""
echo "Build complete! Ready to publish:"
echo "  npm publish --access public"
