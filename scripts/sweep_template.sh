#!/bin/bash
#SBATCH --job-name=itervae_train                   # Job name
#SBATCH --output logs/itervae_train_%J.log         # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<YOUR_EMAIL>                   # Where to send mail
#SBATCH --partition pi_dijk
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb                                 # Job memory request
#SBATCH --time=2-00:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
conda activate itervae
cd <PATH_TO_DIRECTORY>
export CUDA_LAUNCH_BLOCKING=1
echo CUDA_LAUNCH BLOCKING is $CUDA_LAUNCH_BLOCKING
wandb login <YOUR_API_KEY>
export SWEEP_ID=$(wandb sweep sweep.yaml 2>&1 | awk '/with:/{print $8}')
echo SWEEP_ID is $SWEEP_ID
wandb agent $SWEEP_ID