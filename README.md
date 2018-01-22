
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

Use the `train.py` script to train the model.

## Pretrained model checkpoint

You can download our pretrained (TensorFlow) [CIFAR10 model](https://s3.amazonaws.com/temporalfewshot/pixelsnail/cifar.zip) and [ImageNet model](https://s3.amazonaws.com/temporalfewshot/pixelsnail/imagenet.zip)

### CIFAR10
```
python train.py \
       --data_set=cifar \
       --model=h12_noup_smallkey \
       --nr_logistic_mix=10 \
       --nr_filters=256 \
       --batch_size=8 \
       --init_batch_size=8 \
       --dropout_p=0.5 \
       --polyak_decay=0.9995 \
       --save_interval=10
```

### ImageNet
```
python train.py \
       --data_set=imagenet \
       --model=h12_noup_smallkey \
       --nr_logistic_mix=32 \
       --nr_filters=256 \
       --batch_size=8 \
       --init_batch_size=8 \
       --learning_rate=0.001 \
       --dropout_p=0.0 \
       --polyak_decay=0.9997 \
       --save_interval=1
```
