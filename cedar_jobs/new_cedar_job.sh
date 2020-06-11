#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:v100l:1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --mem=128000          # Request the full memory of the node
#SBATCH --time=12:00:00
#SBATCH --account=def-hefeeda
#SBATCH --mail-user=omossad@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH -o /home/omossad/projects/def-hefeeda/omossad/roi_detection/codes/ROI-PyTorch/job.out
#SBATCH -e /home/omossad/projects/def-hefeeda/omossad/roi_detection/codes/ROI-PyTorch/job.err

module load cuda
module load cudnn
module load python
cd /home/omossad/projects/def-hefeeda/omossad/somi/
source venv3/bin/activate
cd /home/omossad/projects/def-hefeeda/omossad/roi_detection/codes/ROI-PyTorch
CUDA_VISIBLE_DEVICES=0 python roi_train.py
