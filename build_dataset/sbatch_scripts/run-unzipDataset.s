#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time32:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=unzip_dataset
#SBATCH --mail-type=END
#SBATCH --mail-user=asl617@nyu.edu
#SBATCH --output=unzip_dataset_%A_%a.out
#SBATCH --error=unzip_dataset_%A_%a.err

module purge
cd '/scratch/$USER/Spring Dataset/'
zip -FFv $SLURM_ARRAY_TASK_ID.zip --out $SLURM_ARRAY_TASK_ID-fixed.zip
rm $SLURM_ARRAY_TASK_ID.zip
mkdir $SLURM_ARRAY_TASK_ID
mv $SLURM_ARRAY_TASK_ID-fixed.zip $SLURM_ARRAY_TASK_ID
cd $SLURM_ARRAY_TASK_ID
unzip $SLURM_ARRAY_TASK_ID-fixed.zip
