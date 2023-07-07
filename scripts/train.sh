#!/bin/bash
#SBATCH --job-name=itervae_train                 # Job name
#SBATCH --output logs/itervae_train_%J.log        # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu                       # Where to send mail
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
conda activate vae
cd /home/dfl32/Desktop/itervae/

python train.py
