#!/bin/bash

# Set input and output directories
DATA_DIR="./data"
INPUT_DIR="$DATA_DIR/CelebV_HQ/videos"
OUTPUT_DIR="$DATA_DIR/CelebV_HQ/images"

# Check if file argument is provided
if [ $# -eq 0 ]; then
    # No argument: get all mp4 files from input directory
    files=$(ls "$INPUT_DIR/"*.mp4 2>/dev/null)
    if [ -z "$files" ]; then
        echo "Error: No .mp4 files found in $INPUT_DIR/"
        exit 1
    fi
else
    # Use provided filename
    files="$INPUT_DIR/$1.mp4"
    # Verify if the specified file exists
    if [ ! -f "$files" ]; then
        echo "Error: File '$1.mp4' does not exist in $INPUT_DIR/"
        exit 1
    fi
fi

# Process each file
for file in $files; do
    # Extract filename without extension
    filename=$(basename "$file" .mp4)

    # Create output directory for this file under images
    mkdir -p "$OUTPUT_DIR/$filename"

    # Define output pattern
    output_pattern="$OUTPUT_DIR/$filename/${filename}_frame%04d.jpg"

    if [ -f "$file" ]; then
        # Extract all frames, starting from 0, with highest quality
        ffmpeg -i "$file" -start_number 0 -q:v 2 "$output_pattern" -hide_banner -loglevel error
        echo "Processed: $filename"
    fi
done

echo "Frame extraction completed for all files."