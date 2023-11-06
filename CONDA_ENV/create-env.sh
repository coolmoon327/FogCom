# 1. create a new env

# 1.1 prepare cuda env
# ubuntu 20.04 with cuda 12.1 for example
# if you have an old version, first uninstall it like: sudo /usr/local/cuda-11.7/bin/cuda-uninstaller
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# 1.2 install python env
conda create -n deeplearning python=3.8
conda activate deeplearning 
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge gym
conda install tensorboardX

# 2. (or) create from spec-list

conda create --name deeplearning --file spec-list.txt