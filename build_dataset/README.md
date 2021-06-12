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

| sbatch filename | description | command to queue job (for scenes 1-7) | prerequisites to running |
| ------------- | ------------- | ------------- | ------------- |
| run-downloadDataset.s | Downloads the dataset to `/scratch/$USER/Dataset/` from shared Google Drive folder, then fixes the downloaded .zip files and unzips them to separate folders. |  `sbatch --array=1-7 run-downloadDataset.s` | Must have empty directory `/scratch/Dataset/` job. Change rclone command in `run-downloadDataset.s` to use your rclone remote name, not `test_4_13_21`. |
| run-colorExtraction.s | In the recording directory level: Creates `Dataset/scene#/recording#/color_frames/` directories, containing extracted color .png frames from each .mkv. Creates an .mp4 video of the frames. | `sbatch --array=1-7 run-colorExtraction.s` | Must have completed `run-downloadDataset.s` |
| run-maskGeneration.s | In the recording directory level: Creates `Dataset/scene#/recording#/hand_frames/masks/` directories, containing low-resolution .png binary masks of each color .png frame using pretrained pixel-level classification models. | `sbatch --array=1-7 run-maskGeneration.s` | Must have completed `run-colorExtraction.s` job. Then download `HandPoseEstimation.py `, `HandPrediction.py`, `HandPredictionModel.py` and place it in each scene directory (i.e. Datset/1/ for scene 1). For each scene, also download the respective `model.pkl` and upload it to that directory level in your scratch space. The python scripts will load this model to predict hand pixels for masking. |
| run-markClipRanges.s | Creates a `clip_ranges.txt` file in each recording's hand_frames directory, containing the frame number ranges for hand appearances/actions in the dataset. | `sbatch --array=1-7 run--markClipRanges.s` | Must have completed `run-maskGeneration.s`. For each scene, download its appropriate `MarkClipRanges.py` script, and upload it to its respective scene directory. |
| run-handPoseEstimation.s | Creates a `hand_landmarks.txt` file in each recording's hand_frames directory, containing the hand pose estimation outputs of Google MediaPipe Hands on each frame in the recording's corresponding `clip_ranges.txt`. | `sbatch --array=1-7 run-handPoseEstimation.s` | Must have completed `run-markClipRanges.s` job. Then download `HandPoseEstimation.py` and place it in each scene directory (i.e. /Dataset/1/ for scene 1). |
| run-zipImgs.s | Zips all masks/ and color_frames/ subdirectories. | `sbatch --array=1-7 run-zipImgs.s` | Must have completed `run-handPoseEstimation.s` job. |
| run-uploadDataset.s | Uploads the dataset. | `sbatch --array=1-7 run-uploadDataset.s` | Must have completed `run-zipImgs.s`. Change rclone command in `run-uploadDataset.s` to use your rclone remote name, not `test_4_13_21`. |
