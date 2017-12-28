
# PixelSNAIL

This is a Python3 / [Tensorflow](https://www.tensorflow.org/) implementation 
of PixelSNAIL.

This code base is based on OpenAI's [PixelCNN++](https://github.com/openai/pixel-cnn) code.

## Setup

To run this code you need the following:

- a machine with multiple GPUs
- Python3
- Numpy, TensorFlow

## Training the model

Use the `train.py` script to train the model. To train the default model on 
CIFAR-10, please refer to `cifar_local.sh` for default parameters.

## Pretrained model checkpoint

You can download our pretrained (TensorFlow) [CIFAR10 model](https://s3.amazonaws.com/temporalfewshot/pixelsnail/cifar.zip) and [ImageNet model](https://s3.amazonaws.com/temporalfewshot/pixelsnail/imagenet.zip)
