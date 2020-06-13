#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --mem=128G            # Request the full memory of the node
#SBATCH --time=12:00:00
#SBATCH --account=def-hefeeda
#SBATCH --mail-user=omossad@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/omossad/projects/def-hefeeda/omossad/roi_detection/codes/ROI-PyTorch/cedar_jobs/job.out
#SBATCH -e /home/omossad/projects/def-hefeeda/omossad/roi_detection/codes/ROI-PyTorch/cedar_jobs/job.err

module load cuda
module load cudnn
module load python
cd /home/omossad/projects/def-hefeeda/omossad/somi/
source venv3/bin/activate
cd /home/omossad/projects/def-hefeeda/omossad/roi_detection/codes/ROI-PyTorch
CUDA_VISIBLE_DEVICES=0 python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/am_0
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/am_2
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_0
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_2
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_4
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_6
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_8
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/kh_0
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/kh_2
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ma_1
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ne_0
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pa_1
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pa_3
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pa_5
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pa_7
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pa_9
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pu_1
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pu_3
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pu_5
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pu_7
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/se_1
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/am_1
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/am_3
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_1
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_3
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_5
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_7
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ha_9
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/kh_1
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ma_0
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/ma_2
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pa_0
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pa_2
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pa_4
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pa_6
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pa_8
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pu_0
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pu_2
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pu_4
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/pu_6
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/se_0
python detect_files.py --image_folder /home/omossad/scratch/Gaming-Dataset/selected_frames/fifa/se_2
