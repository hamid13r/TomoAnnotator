#!/bin/bash
#SBATCH --job-name=organelle_preproc
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/preproc_%j.out
#SBATCH --error=logs/preproc_%j.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load anaconda3   # verify with: module av 2>&1 | grep -i conda

conda activate tomoannotator

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"

python scripts/preprocess.py \
    --input-dir  /gpfs/scratch/$USER/TomoAnnotator/raw/ \
    --output-dir /gpfs/scratch/$USER/TomoAnnotator/processed/

echo "Finished: $(date)"
