#!/bin/bash
# submit_all.sh - submit all scripts in a directory to SLURM via sbatch

DIR="${1:-.}"          # directory to scan, defaults to current dir
PATTERN="${2:-*.sbatch}"   # file pattern, defaults to *.sh

for script in "$DIR"/$PATTERN; do
    if [[ -f "$script" ]]; then
        echo "Submitting: $script"
        sbatch "$script"
    fi
done