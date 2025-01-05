# DeepSMOTE_v2
conda create --name DeepSMOTE python=3.6
conda activate DeepSMOTE


pip install numpy==1.17
conda install cudatoolkit==10.0.130
pip install --upgrade scikit-learn
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html


# DeepSMOTE
DeepSMOTE paper: https://arxiv.org/pdf/2105.02340.pdf

## Code
This repository contains sample code to implement DeepSMOTE.  The first file, DeepSMOTE_MNIST.py, contains code to train a model on the MNIST dataset.  The second file, GenerateSamples, provides code to generate samples on a trained model.

## Data and Pre-Trained Models

Sample training images and labels, as well as saved models are available for download at:
https://drive.google.com/drive/folders/1GRrpiR0CJpcfpjBKO18FLjombxgqH9cK?usp=sharing

## Dependencies

The code was written with: Numpy 1.17; Python 3.6; Pytorch 1.3; and Scikit learn 0.24.1. 


