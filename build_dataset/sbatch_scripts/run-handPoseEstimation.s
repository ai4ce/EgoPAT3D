#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=191GB
#SBATCH --job-name=handPoseEstimation
#SBATCH --mail-type=END
#SBATCH --mail-user=asl617@nyu.edu
#SBATCH --output=handPoseEstimation_%A_%a.out
#SBATCH --output=handPoseEstimation_%A_%a.err

module purge
module load python/intel/3.8.6
pip install --user opencv-contrib-python
pip install --user mediapipe
cd /scratch/$USER/Dataset/$SLURM_ARRAY_TASK_ID
for file in */; do
	hand_frames_subfolder="hand_frames";
	rm $file/$hand_frames_subfolder/hand_landmarks.txt;
	python -B HandPoseEstimation.py "$file";
	done
