#!/bin/bash
#SBATCH --account=kempner_grads
#SBATCH --nodes=1
#SBATCH -n 1 # Number of cores
#SBATCH -p kempner_h100
#SBATCH -t 2-23:00
#SBATCH --job-name=train_nomnom
#SBATCH --output=log/output-%j.log
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --mem=250G # Memory per cpu in MB

source ~/.bashrc
mamba deactivate
mamba activate eco

# module load cudnn
# export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/11.8.0-fasrc01/lib64:$LD_LIBRARY_PATH
# module load gcc/10.2.0-fasrc01

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/
module load cuda/12.4.1-fasrc01
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
module load gcc/12.2.0-fasrc01

# Training on single node
python train_nomnom.py

# Training on multi nodes
