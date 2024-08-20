#!/bin/bash

# Exit on error
set -e

# Define the build directory
BUILD_DIR="build"

# Remove any existing build directory
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create a new build directory
echo "Creating build directory..."
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

# Run CMake to configure the project
echo "Configuring project with CMake..."
cmake ..

# Build the project
echo "Building project..."
cmake --build .


