#!/bin/bash
#SBATCH --job-name=organelle_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# GPU type: --gpus=a100:1 preferred, --gpus=v100:1 as fallback
# Check availability: sinfo -o "%P %G"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load anaconda3

conda activate tomoannotator

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Started:  $(date)"

# Step 1: extract patches (CPU-safe, fast)
python scripts/extract_patches.py \
    --data-dir /gpfs/scratch/$USER/TomoAnnotator/processed/ \
    --output   patches.npz

# Step 2: train patch classifier (GPU)
python scripts/train_patch_classifier.py \
    --patches    patches.npz \
    --output-dir models/ \
    --epochs     50

# Step 3: sync to S3
aws s3 sync models/ s3://scrippsresearch-tomoannotator/models/ --region us-west-2
aws s3 cp  patches.npz s3://scrippsresearch-tomoannotator/training/patches.npz --region us-west-2

echo "Finished: $(date)"
