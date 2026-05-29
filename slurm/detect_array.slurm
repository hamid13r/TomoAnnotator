#!/bin/bash
#SBATCH --job-name=organelle_detect
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --array=0-19            # adjust to (number_of_runs - 1)
#SBATCH --output=logs/detect_%A_%a.out
#SBATCH --error=logs/detect_%A_%a.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

module purge
module load anaconda3

conda activate tomoannotator

DATA_DIR=/gpfs/scratch/$USER/TomoAnnotator/processed
mapfile -t RUNS < <(ls -1 "$DATA_DIR")
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$RUN" ]]; then
    echo "No run for array task $SLURM_ARRAY_TASK_ID — exiting"
    exit 0
fi

echo "Job $SLURM_JOB_ID (task $SLURM_ARRAY_TASK_ID): $RUN  $(date)"

python scripts/detect_features.py \
    --tomogram  "$DATA_DIR/$RUN/tomogram.npy" \
    --model-dir models/ \
    --save-heatmaps

echo "Done: $RUN  $(date)"
