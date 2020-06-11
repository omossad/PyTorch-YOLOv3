#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:v100l:1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
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
CUDA_VISIBLE_DEVICES=0 python new_train.py --model_def config/new_custom.cfg --data_config config/new_custom.data --pretrained_weights weights/yolov3.weights --img_size 608 --multiscale_training False --batch_size 8 --epochs 10
