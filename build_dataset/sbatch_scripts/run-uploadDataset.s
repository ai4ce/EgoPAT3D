#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=uploadDataset
#SBATCH --mail-type=END
#SBATCH --mail-user=asl617@nyu.edu
#SBATCH --output=uploadDataset_%A_%a.out
#SBATCH --output=uploadDataset_%A_%a.err

module purge
module load rclone/1.53.3
gdrive_filepath='AI4CE Research/Egocentric_Intent_Dataset'
scratch_filepath=/scratch/$USER/Dataset
rclone -vv copy "$scratch_filepath/$SLURM_ARRAY_TASK_ID" test_4_13_21:"$gdrive_filepath/$SLURM_ARRAY_TASK_ID"
