#!/bin/bash -login 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
#SBATCH --time=05:00:00
#SBATCH --mem=128000M

module load libs/tensorflow/1.2
module load libs/cudnn/8.0-cuda-8.0
# module load libs/tensorflow/1.2
cd $SLURM_SUBMIT_DIR
echo Running on host `hostname`
# watch -d -n 0.5 nvidia-smi
source ~/tflearn/bin/activate
# python3 convnet_mnist.py

# source ~/dlib/bin/activate
# watch -d -n 0.5 nvidia-smi
# python2 test.py mmod_human_face_detector.dat 56696837.jpg 
# python2 cnn.py
python3 main.py
# source ~/tensorflow/bin/activate
# python3 helloworld.py
