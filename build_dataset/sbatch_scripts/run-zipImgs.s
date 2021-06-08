#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64GB
#SBATCH --job-name=zipIMGs
#SBATCH --mail-type=END
#SBATCH --mail-user=asl617@nyu.edu
#SBATCH --output=zipIMGs_%A_%a.out
#SBATCH --output=zipIMGs_%A_%a.err

module purge
cd /scratch/$USER/Dataset/$SLURM_ARRAY_TASK_ID
for file in */; do
	hand_frames_subfolder="hand_frames";
	color_frames_subfolder="color_frames";
	masks_subfolder="masks";
	pushd $file/$hand_frames_subfolder/;
	zip -r masks.zip $masks_subfolder/;
	rm -r masks;
	popd;
	pushd $file/;
	zip -r color_frames.zip $color_frames_subfolder/;
	rm -r color_frames;
	popd;
	done
