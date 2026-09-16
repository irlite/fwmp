#!/bin/bash
set -e

jid=$(sbatch run_elastic.sbatch | awk '{print $4}')
sbatch --dependency=afterok:$jid make_video.sbatch
