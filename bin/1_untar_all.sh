#!/bin/bash

# Root directory where your .tar files are located
ROOT_DIR="/mnt/ssd/bojing/Image-Adaptive-3DLUT/data/Batch_1"

# Find and extract all .tar files
find "$ROOT_DIR" -type f -name "*.tar" | while read -r tar_file; do
    echo "Processing: $tar_file"

    # Get the directory containing the tar file
    dir=$(dirname "$tar_file")

    # Get the base name (without .tar extension)
    base=$(basename "$tar_file" .tar)

    # Create a folder with the base name for extraction
    extract_dir="$dir/$base"
    mkdir -p "$extract_dir"

    # Extract tar file into that folder
    tar -xf "$tar_file" -C "$extract_dir"

    echo "Extracted to: $extract_dir"

    # Delete the original .tar file
    echo "Deleting: $tar_file"
    rm -f "$tar_file"
done

echo "All .tar files have been extracted."
