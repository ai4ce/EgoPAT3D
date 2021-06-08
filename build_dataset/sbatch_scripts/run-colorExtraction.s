#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=extractColorFrames
#SBATCH --mail-type=END
#SBATCH --mail-user=asl617@nyu.edu
#SBATCH --output=extractColorFrames_%A_%a.out
#SBATCH --output=extractColorFrames_%A_%a.err

module purge
module load ffmpeg/4.2.4
cd /scratch/$USER/Dataset/$SLURM_ARRAY_TASK_ID
for file in *; do
	dir=$(echo $file | cut -d. -f1);
	color_frames_subfolder="color_frames";
	hand_frames_subfolder="hand_frames";
	mkdir -p $dir/$color_frames_subfolder;
	mkdir -p $dir/$hand_frames_subfolder;
	mv $file $dir;
	ffmpeg -i "$dir/$file" -map 0:0 -vsync 0 color%04d.png;
	ffmpeg -r 30 -i color%04d.png -c:v libx264 -vf "fps=30, scale=3840:2160, format=yuv420p" rgb_video.mp4;
	mv "rgb_video.mp4" $dir
	mv color* $dir/$color_frames_subfolder;
	done
