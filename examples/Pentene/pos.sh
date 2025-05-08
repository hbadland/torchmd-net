#!/bin/bash

# Check if the input file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_xyz_file>"
    exit 1
fi

input_file="$1"
output_dir="raw"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Use awk to split the file
awk '
BEGIN { file_count = 0 }
/^      15$/ { 
    if (file_count > 0) {
        close(outfile)
    }
    file_count++
    outfile = sprintf("'$output_dir'/part_%06d.pos", file_count)
}
{ print > outfile }
' "$input_file"

echo "Split $file_count XYZ files into $output_dir"
