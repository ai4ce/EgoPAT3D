Egocentric Intent Dataset
===========
### Useful resources for learning NYU HPC usage  
[NYU HPC eligibility and access](https://www.nyu.edu/life/information-technology/research-and-data-support/high-performance-computing/high-performance-computing-accounts.html)  

Recommended resource:  
[NYU HPC guide covering the scope of concepts used for the Egocentric Intent Dataset](https://docs.google.com/document/d/1gnW7C9B5QVSTrQ8s0TPMIE7UgkXThcK5H-TX-6OBK44/edit)  
[Official NYU HPC Basics Video Series](https://www.youtube.com/watch?v=0pP_TeKH1MI&list=PL5l6Qz3Xhfi9Jn9-iMKJisYsSW5tRzPSd)  

Most comprehensive resource:  
[Official NYU HPC Documentation / Guides](https://sites.google.com/a/nyu.edu/nyu-hpc/documentation?authuser=0)  

## Manual Build Instructions using NYU HPC (Greene Supercomputer)
1. Download the `.s` sbatch files found in this repository's `sbatch_scripts/` directoy
2. Connect to NYU Greene
3. Configure rclone on your HPC account  
[NYU HPC rclone documentation](https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/data-management/transfering-data/google-drive) + ["How to transfer files from Google Drive to your scratch folder" section from dataset HPC guide](https://docs.google.com/document/d/1gnW7C9B5QVSTrQ8s0TPMIE7UgkXThcK5H-TX-6OBK44/edit)
4. Edit several sbatch files to fit your configuration:  
`run-downloadDataset.s`, `run-uploadDataset.s`: change rclone commands to use your rclone remote name, not `test_4_13_21`.  
all sbatch files: change `#SBATCH --mail-user=asl617@nyu.edu` to use your email address for slurm job status notifications.  
5. Navigate to your scratch space in Greene  
`cd /scratch/$USER` where $USER is your netID, i.e. `cd /scratch/asl617`  

In your scratch space,  

6. Upload the sbatch files  
7. Create a new directory called `Dataset/` :  
`mkdir Dataset`  
8. Start queuing slurm jobs in the order below. Do not queue the next slurm job before the previous one is 100% done running and you have met the prerequisites specified below (`squeue -u $USER` to check job statuses).  

| sbatch filename | description | command to queue job |
| ------------- | ------------- |
| run-downloadDataset.s |  | |
| 2 |  |
| 3 |  |
| 4 |  |
| 5 |  |
| 6 |  |
| 7 | kitchenCupboard |
| 8 | kitchenSink |
| 9 | microwave |
| 10 | nightstand |
| 11 | pantryShelf |
| 12 | smallBins |
| 13 | stoveTop |
| 14 | windowsillAC |
| 15 | woodenTable |
