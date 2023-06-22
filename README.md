# Iterative VAE

This repo trains and compares iterative VAEs against traditional VAEs. Support for more datasets, models, and command-line argument parsing to come. 

To get started, create a conda environment:
```
conda create -n itervae
conda activate itervae
```
Install the following packages:
```
conda install matplotlib tqdm
```
Select the right pytorch installation from the pytorch website [here](https://pytorch.org/get-started/locally/) and install using conda.

## Data

Training is done for the MNIST, EMNIST, and CIFAR10 datasets.

## Models

Models are trained with the following hyperparameters:
- 100 epochs
- Non-iterative decoder with 1 hidden layer
- Non-iterative encoder with 1-5 hidden layers or an encoder with 1 hidden layer encoder iterated 1-5 times
- Hidden dimension of either 200 or 400
- Latent dimension of 200

## Plots

4 loss plots over epochs are generated for each (hidden_dim, dataset) pair:
1. Non-iterative training loss for all numbers of layers
2. Non-iterative validation loss for all numbers of layers
3. Iterative training loss for all numbers of iterations
4. Iterative validation loss for all numbers of iterations

These plots are saved in the ```plots``` folder.

## Train

To train and generate plots, run the following command in your terminal:

```
python train.py
```