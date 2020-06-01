# PyTorch-YOLOv3
Implementation of ROI detection using PyTorch

## Installation
##### Clone and install requirements
    $ git clone https://github.com/omossad/ROI-PyTorch.git
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
  Pretrained weights are available under
    /home/omossad/scratch/ROI-PyTorch/checkpoints/tiny_yolo.pth

##### Dataset
  2 versions of the dataset are available:
  train-dev.txt or train.txt
  variable size for testing or full trial

##### Quick running
python roi_train.py

##### File structure
roi_train.py : the file including the training iterations and commands
roi_model.py : the file containing the model implementation and details.
