#!/bin/bash

# Set input and output directories
DATA_DIR="./data"
INPUT_DIR="$DATA_DIR/MEAD/W024/video"
OUTPUT_DIR="$DATA_DIR/MEAD/W024/images"

# Default values (if not provided via arguments)
if [ $# -eq 0 ]; then
    # No arguments: get all emotions from front directory
    EMOTIONS=($(ls -d "$INPUT_DIR/front/"*/ | xargs -n 1 basename))
else
    # Use provided emotion
    EMOTIONS=("$1")
    # Verify if the specified emotion exists
    if [ ! -d "$INPUT_DIR/front/$1" ]; then
        echo "Error: Emotion '$1' does not exist in $INPUT_DIR/front/"
        exit 1
    fi
fi

# Process each emotion
for emotion in "${EMOTIONS[@]}"; do
    # Get levels based on arguments or directory listing
    if [ $# -le 1 ]; then
        # No level specified: get all levels
        LEVELS=($(ls -d "$INPUT_DIR/front/$emotion/"*/ | xargs -n 1 basename))
    else
        # Use provided level
        LEVELS=("$2")
        # Verify if the specified level exists
        if [ ! -d "$INPUT_DIR/front/$emotion/$2" ]; then
            echo "Error: Level '$2' does not exist in $INPUT_DIR/front/$emotion/"
            exit 1
        fi
    fi

    for level in "${LEVELS[@]}"; do
        # Get filenames based on arguments or directory listing
        if [ $# -le 2 ]; then
            # No filename specified: get all mp4 files
            files=$(ls "$INPUT_DIR/front/$emotion/$level/"*.mp4 2>/dev/null)
            if [ -z "$files" ]; then
                continue
            fi
        else
            # Use provided filename
            files="$INPUT_DIR/front/$emotion/$level/$3.mp4"
            # Verify if the specified file exists
            if [ ! -f "$files" ]; then
                echo "Error: File '$3.mp4' does not exist in $INPUT_DIR/front/$emotion/$level/"
                exit 1
            fi
        fi

        for file in $files; do
            # Extract filename without extension
            filename=$(basename "$file" .mp4)

            # Create output directory
            mkdir -p "$OUTPUT_DIR/$emotion/$level/$filename"

            # Get all angle directories
            ANGLES=($(ls -d "$INPUT_DIR/"*/ | xargs -n 1 basename))

            for angle in "${ANGLES[@]}"; do
                input_file="$INPUT_DIR/$angle/$emotion/$level/$filename.mp4"
                output_pattern="$OUTPUT_DIR/$emotion/$level/$filename/img_%04d_${angle}.jpg"

                if [ -f "$input_file" ]; then
                    # Extract all frames, starting from 0
                    ffmpeg -i "$input_file" -start_number 0 -q:v 2 "$output_pattern" -hide_banner -loglevel error
                    echo "Processed: $emotion/$level/$filename/$angle"
                fi
            done
        done
    done
done

echo "Frame extraction completed for all files."