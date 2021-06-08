#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=markClipRanges
#SBATCH --mail-type=END
#SBATCH --mail-user=asl617@nyu.edu
#SBATCH --output=markClipRanges_%A_%a.out
#SBATCH --output=markClipRanges_%A_%a.err

module purge
module load python/intel/3.8.6	
cd /scratch/$USER/Dataset/$SLURM_ARRAY_TASK_ID
for file in */; do
	hand_frames_subfolder="hand_frames";
	rm $file/$hand_frames_subfolder/*.txt
	python -B MarkClipRanges.py "$file";
	done