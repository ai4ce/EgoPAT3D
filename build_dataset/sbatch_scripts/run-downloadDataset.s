#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=downloadDataset
#SBATCH --mail-type=END
#SBATCH --mail-user=asl617@nyu.edu
#SBATCH --output=downloadDataset_%A_%a.out
#SBATCH --output=downloadDataset_%A_%a.err

module purge
module load rclone/1.53.3
filepath='AI4CE Research/Fall 2020 Action Prediction/Spring Dataset/'
rclone -vv copy test_4_13_21:$filepath$SLURM_ARRAY_TASK_ID.zip /scratch/$USER/Dataset
cd /scratch/$USER/Dataset
zip -FFv $SLURM_ARRAY_TASK_ID.zip --out $SLURM_ARRAY_TASK_ID-fixed.zip
rm $SLURM_ARRAY_TASK_ID.zip
mkdir $SLURM_ARRAY_TASK_ID
mv $SLURM_ARRAY_TASK_ID-fixed.zip $SLURM_ARRAY_TASK_ID
cd $SLURM_ARRAY_TASK_ID
unzip $SLURM_ARRAY_TASK_ID-fixed.zip
rm $SLURM_ARRAY_TASK_ID-fixed.zip
