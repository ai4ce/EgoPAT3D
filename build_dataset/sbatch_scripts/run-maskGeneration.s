#!/bin/bash
#
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=72:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=maskGeneration
#SBATCH --mail-type=END
#SBATCH --mail-user=asl617@nyu.edu
#SBATCH --output=maskGeneration_%A_%a.out
#SBATCH --output=maskGeneration_%A_%a.err

module purge
module load python/intel/3.8.6
cd /scratch/$USER/Dataset/$SLURM_ARRAY_TASK_ID
for file in */; do
	hand_frames_subfolder="hand_frames";
	masks_subfolder="masks";
	mkdir $file/$hand_frames_subfolder/$masks_subfolder;
	python -B HandPrediction.py "$file";
	done
