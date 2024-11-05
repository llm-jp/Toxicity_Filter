#!/bin/bash

# Usage:
# ./parallel_classify.sh <jsonl_directory> <score_directory> <toxic_directory> <nontoxic_directory> <num_cpus> [threshold]

# Function to display usage instructions
usage() {
    echo "Usage: $0 <jsonl_directory> <score_directory> <toxic_directory> <nontoxic_directory> <num_cpus> [threshold]"
    echo
    echo "Positional arguments:"
    echo "  jsonl_directory      Directory containing .jsonl.gz files"
    echo "  score_directory      Directory containing score .txt files corresponding to JSONL files"
    echo "  toxic_directory      Directory to save toxic JSONL files"
    echo "  nontoxic_directory   Directory to save non-toxic JSONL files"
    echo "  num_cpus             Number of CPU cores to utilize for parallel processing"
    echo
    echo "Optional arguments:"
    echo "  threshold            Threshold for classifying toxicity (default: 8.4)"
    echo
    echo "Example:"
    echo "  $0 /data/jsonl /data/scores /data/toxic /data/nontoxic 4 9.0"
    exit 1
}

# Check for minimum number of arguments
if [ "$#" -lt 5 ]; then
    echo "Error: Insufficient arguments provided."
    usage
fi

# Assign positional arguments
JSONL_DIR="$1"
SCORE_DIR="$2"
TOXIC_DIR="$3"
NONTOXIC_DIR="$4"
NUM_CPUS="$5"
THRESHOLD="${6:-8.4}"  # Default to 8.4 if not provided

# Validate that NUM_CPUS is a positive integer
if ! [[ "$NUM_CPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: <num_cpus> must be a positive integer."
    usage
fi

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel is not installed. Please install it and try again."
    exit 1
fi

# Export variables for GNU parallel
export JSONL_DIR
export SCORE_DIR
export TOXIC_DIR
export NONTOXIC_DIR
export THRESHOLD

# Function to process a single file using the Python script
process_single_file() {
    local file="$1"
    python classify_jsonl.py "$JSONL_DIR" "$SCORE_DIR" "$TOXIC_DIR" "$NONTOXIC_DIR" --threshold "$THRESHOLD" --file "$file"
}

export -f process_single_file

# Find all .jsonl.gz files in the JSONL_DIR and pass them to GNU parallel
find "$JSONL_DIR" -maxdepth 1 -type f -name '*.jsonl.gz' | \
    parallel --jobs "$NUM_CPUS" --progress process_single_file {/}

